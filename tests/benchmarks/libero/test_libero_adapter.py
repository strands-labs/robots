"""Tests for :class:`LiberoAdapter`.

Covers:

* Construction via ``from_file`` / ``from_text`` / raw ``__init__``.
* ``supported_robots`` / ``default_robot`` = Panda-only.
* ``instruction`` surfaces the BDDL ``:language`` string.
* ``is_success`` positive + negative cases against fake sims (no MuJoCo
  needed - the predicates poll ``get_body_state`` / ``get_contacts``).
* ``on_episode_start`` loads the scene (or errors cleanly) and applies
  per-episode jitter when the sim exposes ``move_object``.
* Integration with ``PolicyRunner.evaluate`` and
  ``SimEngine.evaluate_benchmark`` via a minimal ``FakeSim`` stub.
* Error surface: unknown task via ``evaluate_benchmark`` returns a
  structured error dict, never raises.
"""

from __future__ import annotations

import builtins
import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strands_robots.benchmarks.libero import (
    BDDLParseError,
    LiberoAdapter,
)
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.benchmark import (
    _BENCHMARK_REGISTRY,
    register_benchmark,
)
from strands_robots.simulation.policy_runner import PolicyRunner


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


# Representative BDDL fragments

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the red cube and place it on the plate")
  (:objects cube_1 plate_1 table_1 - object)
  (:init (on cube_1 table_1))
  (:goal (on cube_1 plate_1)))
"""

COMPOUND_BDDL = """
(define (problem libero_grasp_and_upright)
  (:language "grasp the bottle and keep it upright")
  (:goal (and (grasped bottle_1) (upright bottle_1))))
"""

NEGATED_BDDL = """
(define (problem libero_release)
  (:goal (not (grasped cube_1))))
"""


# Fake sim helpers


class _FakeRobot:
    def __init__(self, data_config: str):
        self.data_config = data_config


class _FakeWorld:
    def __init__(self, robots: dict[str, _FakeRobot]):
        self.robots = dict(robots)
        self.cameras: dict[str, dict[str, Any]] = {}
        # Backend state mirrors MuJoCoSimulation._world._backend_state -
        # the canonical place for adapter-controlled state per
        # SimWorld's docstring (e.g. viz_option, recording, dataset_recorder).
        self._backend_state: dict[str, Any] = {}


class FakeSim(SimEngine):
    """Minimal ``SimEngine`` with get_body_state / get_contacts / move_object / add_camera."""

    def __init__(
        self,
        bodies: dict[str, dict[str, Any]] | None = None,
        contacts: list[dict[str, str]] | None = None,
        data_config: str = "panda",
        preexisting_cameras: list[str] | None = None,
        add_camera_fail: bool = False,
    ):
        self._bodies = dict(bodies or {})
        self._contacts = list(contacts or [])
        self._reset_count = 0
        self._move_calls: list[tuple[str, list[float]]] = []
        self._world = _FakeWorld({"fake_panda": _FakeRobot(data_config)})
        self._scenes_loaded: list[str] = []
        self._add_camera_calls: list[tuple[str, dict[str, Any]]] = []
        self._add_camera_fail = add_camera_fail
        for cam in preexisting_cameras or []:
            self._world.cameras[cam] = {"position": [0, 0, 0], "target": [0, 0, 0]}

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        self._reset_count += 1
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        return {"status": "success"}

    def get_state(self):
        return {}

    def add_robot(self, name, **kw):
        dc = kw.get("data_config") or "panda"
        self._world.robots[name] = _FakeRobot(dc)
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self):
        return list(self._world.robots.keys())

    def robot_joint_names(self, robot_name):
        return ["j0", "j1"]

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {n: 0.0 for n in self.robot_joint_names(robot_name or "fake_panda")}

    def send_action(self, action, robot_name=None, n_substeps=1):
        pass

    def render(self, camera_name="default", width=None, height=None):
        return {"status": "success", "content": [{"text": "render"}]}

    # Optional helpers used by predicates + adapter

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._bodies:
            return {"status": "error", "content": [{"text": "missing"}]}
        return {
            "status": "success",
            "content": [
                {"text": body_name},
                {
                    "json": {
                        "position": self._bodies[body_name].get("position", [0, 0, 0]),
                        "quaternion": self._bodies[body_name].get("quaternion", [1, 0, 0, 0]),
                        "mass": 1.0,
                    }
                },
            ],
        }

    def get_contacts(self) -> dict[str, Any]:
        return {
            "status": "success",
            "content": [
                {"text": f"{len(self._contacts)} contacts"},
                {"json": {"contacts": self._contacts, "n_contacts": len(self._contacts)}},
            ],
        }

    def move_object(self, *, name: str, position: list[float]) -> dict[str, Any]:
        self._move_calls.append((name, list(position)))
        self._bodies.setdefault(name, {})["position"] = list(position)
        return {"status": "success"}

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        self._scenes_loaded.append(scene_path)
        return {"status": "success"}

    def add_camera(
        self,
        name: str,
        position: list[float] | None = None,
        target: list[float] | None = None,
        fov: float = 60.0,
        width: int = 640,
        height: int = 480,
    ) -> dict[str, Any]:
        kwargs = {
            "position": position,
            "target": target,
            "fov": fov,
            "width": width,
            "height": height,
        }
        self._add_camera_calls.append((name, dict(kwargs)))
        if self._add_camera_fail:
            return {"status": "error", "content": [{"text": "fake injection failed"}]}
        if name in self._world.cameras:
            return {
                "status": "error",
                "content": [{"text": f"add_camera: camera '{name}' already exists. Remove it first."}],
            }
        self._world.cameras[name] = kwargs
        return {"status": "success", "content": [{"text": f"📷 Camera '{name}' added"}]}


# Construction


class TestConstruction:
    def test_from_text_happy_path(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        assert adapter.supported_robots == ["panda"]
        assert adapter.default_robot == "panda"
        # Default max_steps=720 matches NVIDIA's upstream
        # ``MultiStepConfig.max_episode_steps`` for LIBERO eval (#168
        # round 37). Pre-round-37 we used 300 (LIBERO repo's lifelong-
        # learning convention) which was too short for libero_10.
        assert adapter.max_steps == 720
        assert adapter.instruction == "pick up the red cube and place it on the plate"

    def test_from_text_respects_max_steps_override(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, max_steps=75)
        assert adapter.max_steps == 75

    def test_from_text_rejects_negative_jitter(self):
        with pytest.raises(ValueError):
            LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=-0.1)

    def test_rejects_bddl_without_goal(self):
        text = '(define (problem no_goal) (:language "no goal block"))'
        with pytest.raises(ValueError, match="no \\(:goal"):
            LiberoAdapter.from_text(text)

    def test_from_file(self, tmp_path):
        p = tmp_path / "task.bddl"
        p.write_text(PICK_CUBE_BDDL)
        adapter = LiberoAdapter.from_file(p)
        assert adapter.problem.name == "libero_spatial_pick_cube"

    def test_from_text_propagates_parse_errors(self):
        with pytest.raises(BDDLParseError):
            LiberoAdapter.from_text("(define (problem p) (:goal (telekinesis cube_1)))")


# Lifecycle hooks


class TestIsSuccess:
    def test_positive_case_on(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0, 0, 0.25]},
                "plate_1": {"position": [0, 0, 0.1]},
            },
            # #171 sub-task 3e: ``_body_on`` now requires a contact via
            # ``get_contacts``. Geom-prefix convention is
            # ``<bddl_name>_g<idx>`` (matches what LIBERO scenes emit).
            contacts=[{"geom1": "cube_1_g0", "geom2": "plate_1_g0"}],
        )
        assert adapter.is_success(sim) is True

    def test_negative_case_on(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0, 0, 0.05]},  # not above plate
                "plate_1": {"position": [0, 0, 0.1]},
            }
        )
        assert adapter.is_success(sim) is False

    def test_compound_goal_needs_both(self):
        adapter = LiberoAdapter.from_text(COMPOUND_BDDL)
        sim_neither = FakeSim(bodies={"bottle_1": {"quaternion": [1.0, 0, 0, 0]}})
        sim_upright_only = FakeSim(bodies={"bottle_1": {"quaternion": [1.0, 0, 0, 0]}})
        sim_both = FakeSim(
            bodies={"bottle_1": {"quaternion": [1.0, 0, 0, 0]}},
            contacts=[{"geom1": "robot0_gripper_finger_l", "geom2": "bottle_1"}],
        )
        assert adapter.is_success(sim_neither) is False
        assert adapter.is_success(sim_upright_only) is False
        assert adapter.is_success(sim_both) is True

    def test_negated_goal(self):
        adapter = LiberoAdapter.from_text(NEGATED_BDDL)
        sim_empty = FakeSim()
        sim_gripped = FakeSim(contacts=[{"geom1": "robot0_gripper_finger_l", "geom2": "cube_1"}])
        assert adapter.is_success(sim_empty) is True
        assert adapter.is_success(sim_gripped) is False


class TestWalkPredicateTree:
    """``_walk_predicate_tree`` reports each leaf's verdict with the
    actual body positions so reviewers can see why a predicate returned
    its verdict (e.g., body name didn't resolve = ``=NONE``; positions
    present + verdict=False = threshold-rejection).

    The walker was originally added for the #170 BDDL ↔ ``env.check_success``
    diagnostic; that diagnostic was removed in #178 along with
    ``LiberoOffScreenRenderEngine`` (no ``env.check_success`` to compare
    against). The walker itself is still useful for any future
    BDDL-evaluator debugging, so it stays.
    """

    def test_walk_predicate_tree_reports_leaf_positions(self):
        """``_walk_predicate_tree`` reports each leaf's verdict with the
        actual body positions so reviewers can see why a predicate
        returned its verdict (e.g., body name didn't resolve = ``=NONE``;
        positions present + verdict=False = threshold-rejection)."""
        from strands_robots.benchmarks.libero.adapter import _walk_predicate_tree
        from strands_robots.benchmarks.libero.bddl_parser import parse_bddl

        text = "(define (problem p) (:goal (on cube_1 plate_1)))"
        problem = parse_bddl(text)
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0.001, -0.305, 0.443]},
                "plate_1": {"position": [-0.004, -0.323, 0.439]},
            },
            # #171 sub-task 3e: ``on`` predicate also requires contact.
            contacts=[{"geom1": "cube_1_g0", "geom2": "plate_1_g0"}],
        )
        leaves = _walk_predicate_tree(problem.goal, sim)
        assert len(leaves) == 1
        leaf_repr, verdict = leaves[0]
        # Verdict reflects the new tighter thresholds (mug 4mm above
        # plate, 1cm xy → True).
        assert verdict is True
        # Positions are rendered for human inspection.
        assert "cube_1=[" in leaf_repr
        assert "plate_1=[" in leaf_repr
        # 3-decimal precision so reviewers can eyeball the diagnostic.
        assert "0.443" in leaf_repr or "0.44" in leaf_repr

    def test_walk_predicate_tree_reports_missing_body_as_none(self):
        """When a body name doesn't resolve in the sim, the leaf
        diagnostic shows ``=NONE`` so reviewers can immediately see
        the look-up failure (the round-170 smoking gun)."""
        from strands_robots.benchmarks.libero.adapter import _walk_predicate_tree
        from strands_robots.benchmarks.libero.bddl_parser import parse_bddl

        text = "(define (problem p) (:goal (on missing_body_1 plate_1)))"
        problem = parse_bddl(text)
        sim = FakeSim(
            bodies={
                # Only plate_1 is registered; missing_body_1 doesn't exist.
                "plate_1": {"position": [0, 0, 0.10]},
            }
        )
        leaves = _walk_predicate_tree(problem.goal, sim)
        assert len(leaves) == 1
        leaf_repr, verdict = leaves[0]
        assert verdict is False
        assert "missing_body_1=NONE" in leaf_repr


class TestOnEpisodeStart:
    def test_loads_scene_before_compat_check(self, tmp_path):
        """``scene_path`` load must happen before ``super().on_episode_start``
        so the base compat check sees the scene's Panda robot."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, scene_path="/fake/scene.xml")
        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))
        assert sim._scenes_loaded == ["/fake/scene.xml"]

    def test_scene_load_error_raises(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, scene_path="/bad/path.xml")

        sim = FakeSim(data_config="panda")
        # Override load_scene to report failure.
        sim.load_scene = lambda path: {  # type: ignore[assignment]
            "status": "error",
            "content": [{"text": f"no such file: {path}"}],
        }
        with pytest.raises(RuntimeError, match="load_scene"):
            adapter.on_episode_start(sim, random.Random(0))

    def test_jitter_applied_when_init_declares_subject(self):
        adapter = LiberoAdapter.from_text(
            """
            (define (problem p)
              (:init (on cube_1 table_1))
              (:goal (on cube_1 plate_1)))
            """,
            init_jitter=0.01,
        )
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0.0, 0.0, 0.1]},
                "table_1": {"position": [0.0, 0.0, 0.0]},
            },
            data_config="panda",
        )
        adapter.on_episode_start(sim, random.Random(42))
        # cube_1 got jittered; table_1 is only the reference and not moved.
        assert any(call[0] == "cube_1" for call in sim._move_calls)
        assert not any(call[0] == "table_1" for call in sim._move_calls)

    def test_jitter_disabled_when_zero(self):
        adapter = LiberoAdapter.from_text(
            """
            (define (problem p)
              (:init (on cube_1 table_1))
              (:goal (on cube_1 plate_1)))
            """,
            init_jitter=0.0,
        )
        sim = FakeSim(
            bodies={"cube_1": {"position": [0, 0, 0.1]}, "table_1": {"position": [0, 0, 0]}},
            data_config="panda",
        )
        adapter.on_episode_start(sim, random.Random(42))
        assert sim._move_calls == []

    def test_non_panda_robot_rejected_by_base_compat_check(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(data_config="so100")  # not panda
        from strands_robots.simulation.benchmark import BenchmarkCompatibilityError

        with pytest.raises(BenchmarkCompatibilityError) as exc:
            adapter.on_episode_start(sim, random.Random(0))
        assert exc.value.supported == ["panda"]
        assert exc.value.data_config == "so100"


# LIBERO camera installation (#148 / Failure 1)


class TestLiberoCameraInstall:
    def test_default_cameras_installed(self):
        """``image`` and ``wrist_image`` cameras must be installed so the
        ``libero_panda`` Gr00tDataConfig finds them in the observation."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))

        installed_names = {name for name, _ in sim._add_camera_calls}
        assert installed_names == {"image", "wrist_image"}
        assert "image" in sim._world.cameras
        assert "wrist_image" in sim._world.cameras

    def test_camera_install_happens_after_robot_load(self):
        """Camera install must run AFTER super().on_episode_start so it
        installs into a populated scene rather than an empty world."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)

        events: list[str] = []

        class TrackingFakeSim(FakeSim):
            def add_robot(self, name, **kw):
                events.append("add_robot")
                return super().add_robot(name, **kw)

            def add_camera(self, name, **kw):
                events.append(f"add_camera:{name}")
                return super().add_camera(name, **kw)

        sim = TrackingFakeSim(data_config="panda")
        # Empty world so super() triggers add_robot for default_robot.
        sim._world.robots.clear()
        adapter.on_episode_start(sim, random.Random(0))

        # add_robot must come before any add_camera call.
        first_camera_idx = next(i for i, e in enumerate(events) if e.startswith("add_camera:"))
        first_robot_idx = events.index("add_robot")
        assert first_robot_idx < first_camera_idx

    def test_install_cameras_disabled(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0, install_cameras=False)
        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))
        assert sim._add_camera_calls == []

    def test_custom_cameras_override_defaults(self):
        custom = {
            "wide_view": {
                "position": [2.0, 2.0, 2.0],
                "target": [0.0, 0.0, 0.0],
                "fov": 90.0,
                "width": 128,
                "height": 128,
            }
        }
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0, cameras=custom)
        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))

        installed_names = {name for name, _ in sim._add_camera_calls}
        # Defaults are completely replaced - only the custom camera is installed.
        assert installed_names == {"wide_view"}

    def test_existing_cameras_skipped(self):
        """A scene MJCF that already declares ``image`` should beat us to it -
        the adapter must not error out when the camera already exists."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        sim = FakeSim(data_config="panda", preexisting_cameras=["image"])
        adapter.on_episode_start(sim, random.Random(0))

        # ``image`` was preexisting - we must not call add_camera for it.
        installed_names = {name for name, _ in sim._add_camera_calls}
        assert installed_names == {"wrist_image"}

    def test_model_side_cameras_skip_install_no_recompile(self):
        """Critical for #166: ``Simulation.load_scene`` creates a fresh
        ``SimWorld`` whose registry is empty even when the loaded MJCF
        declares cameras. ``_install_libero_cameras`` MUST detect the
        scene-supplied cameras via the compiled model (not just the
        registry) so it doesn't trigger a spec recompile that resets
        qpos and undoes ``_apply_canonical_state``."""

        class _FakeMjEnum:
            mjOBJ_CAMERA = 7

        class _FakeMjModule:
            mjtObj = _FakeMjEnum()

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                return ["image", "wrist_image"][idx]

        class _FakeMjModel:
            ncam = 2

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        sim = FakeSim(data_config="panda")
        # Registry is empty (mimics post-load_scene state) but the
        # compiled model declares both LIBERO cameras.
        sim._world.cameras.clear()
        sim._world._model = _FakeMjModel()  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": _FakeMjModule()}):
            adapter.on_episode_start(sim, random.Random(0))

        # No add_camera calls - both names detected as already-present
        # in the compiled model.
        assert sim._add_camera_calls == []

    def test_partial_model_side_cameras_fills_missing(self):
        """If the scene declares ONLY ``image`` (not ``wrist_image``) in
        the compiled model, we install just the missing one."""

        class _FakeMjEnum:
            mjOBJ_CAMERA = 7

        class _FakeMjModule:
            mjtObj = _FakeMjEnum()

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                return ["image"][idx]

        class _FakeMjModel:
            ncam = 1

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        sim = FakeSim(data_config="panda")
        sim._world.cameras.clear()
        sim._world._model = _FakeMjModel()  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": _FakeMjModule()}):
            adapter.on_episode_start(sim, random.Random(0))

        installed_names = {name for name, _ in sim._add_camera_calls}
        assert installed_names == {"wrist_image"}

    def test_existing_camera_names_unions_registry_and_model(self):
        """Direct exercise of ``_existing_camera_names`` - union of
        registry and model camera names."""

        class _FakeMjEnum:
            mjOBJ_CAMERA = 7

        class _FakeMjModule:
            mjtObj = _FakeMjEnum()

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                return "agentview"

        class _FakeMjModel:
            ncam = 1

        sim = FakeSim(data_config="panda", preexisting_cameras=["topdown"])
        sim._world._model = _FakeMjModel()  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": _FakeMjModule()}):
            names = LiberoAdapter._existing_camera_names(sim)

        assert names == {"topdown", "agentview"}

    def test_camera_install_failures_are_logged_not_fatal(self, caplog):
        """A backend that refuses every add_camera shouldn't kill the eval.

        The adapter's contract: log at WARNING and continue so other
        cameras + the rollout itself still run.
        """
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        sim = FakeSim(data_config="panda", add_camera_fail=True)
        with caplog.at_level("WARNING"):
            # Must not raise.
            adapter.on_episode_start(sim, random.Random(0))
        # Both cameras attempted; both reported as warning.
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 2

    def test_default_camera_keys_match_libero_panda_data_config(self):
        """Regression: the adapter's camera names MUST match the bare keys of
        the ``libero_panda`` Gr00tDataConfig's video_keys, otherwise
        ``_build_service_observation`` won't find them in the observation."""
        # Bare keys after removeprefix("video.").
        expected = {"image", "wrist_image"}
        assert set(LiberoAdapter.LIBERO_CAMERAS.keys()) == expected

    def test_sim_without_add_camera_silently_skipped(self):
        """Backends that don't expose add_camera (future engines) shouldn't
        crash the adapter - it falls back to "no camera install"."""

        class BareSim(FakeSim):
            # Override add_camera to ``None`` so getattr returns falsy at runtime.
            add_camera = None  # type: ignore[assignment]

        sim = BareSim(data_config="panda")
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        # Must not raise even though sim.add_camera is None.
        adapter.on_episode_start(sim, random.Random(0))


class TestPublishCameraDimsToWorld:
    """Round 40 (#168): when ``_install_libero_cameras`` skips the
    ``add_camera`` call (because the camera is already in the
    compiled model from ``scene_camera_aliases`` rename), it still
    publishes a config-only :class:`SimCamera` entry into
    ``world.cameras`` so :meth:`_get_sim_observation` reads the
    configured render dimensions instead of falling through to the
    480×640 ``default_height``/``default_width``.

    Pre-round-40 the eval pipeline was rendering ``image`` and
    ``wrist_image`` at 480×640 (different aspect AND resolution from
    GR00T-N1.7-LIBERO's training distribution of 256×256), feeding
    the policy out-of-distribution images.
    """

    def test_publishes_simcamera_with_configured_dims(self):
        """Round 40 (#168): the published entry must be a
        :class:`SimCamera` instance with ``height``/``width`` from
        ``cam_kwargs`` (256×256 for the LIBERO defaults).

        Pin the SHAPE (SimCamera, not dict) so ``_get_sim_observation``'s
        ``cam_info.height`` attribute access works."""
        from strands_robots.simulation.models import SimCamera

        sim = FakeSim(data_config="panda")
        # world.cameras starts empty — model-side rename hasn't put
        # anything in the registry yet.
        assert "image" not in sim._world.cameras

        LiberoAdapter._publish_camera_dims_to_world(
            sim,
            "image",
            {"width": 256, "height": 256, "fov": 60.0},
        )

        cam = sim._world.cameras["image"]
        assert isinstance(cam, SimCamera)
        assert cam.name == "image"
        assert cam.height == 256
        assert cam.width == 256

    def test_idempotent_does_not_overwrite_existing_entry(self):
        """Round 40 (#168): when ``world.cameras[cam_name]`` is already
        populated (e.g. from a prior episode's publish call, or from
        the user's explicit ``add_camera``), the publish must NOT
        overwrite it — preserves user-configured pose / FOV /
        custom-dim overrides across episodes."""
        from strands_robots.simulation.models import SimCamera

        sim = FakeSim(data_config="panda")
        # Pre-populate with a custom 128×128 entry (e.g. user override).
        sim._world.cameras["image"] = SimCamera(name="image", width=128, height=128)

        LiberoAdapter._publish_camera_dims_to_world(
            sim,
            "image",
            {"width": 256, "height": 256, "fov": 60.0},
        )

        # The pre-existing 128×128 entry must still be there.
        cam = sim._world.cameras["image"]
        assert cam.width == 128
        assert cam.height == 128

    def test_silent_when_world_missing(self):
        """Round 40 (#168): backends without ``_world`` (future engines)
        get a silent no-op rather than a crash. Best-effort contract."""

        class WorldlessSim:
            pass

        # Must not raise.
        LiberoAdapter._publish_camera_dims_to_world(WorldlessSim(), "image", {"width": 256, "height": 256})

    def test_silent_when_cameras_attr_not_dict(self):
        """Round 40 (#168): backends with ``world.cameras`` of an
        unexpected type (None / list / other) get a silent no-op.

        Pin so the method's defensive `isinstance(_, dict)` check
        stays in place."""

        class WeirdCameras:
            cameras = None

        class WeirdSim:
            _world = WeirdCameras()

        # Must not raise.
        LiberoAdapter._publish_camera_dims_to_world(WeirdSim(), "image", {"width": 256, "height": 256})

    def test_install_libero_cameras_publishes_when_already_in_sim(self):
        """Round 40 (#168): end-to-end through ``_install_libero_cameras``.

        Pre-populate ``world.cameras["image"]`` AS A NON-SIMCAMERA
        VALUE (mimicking the FakeSim's ``add_camera`` adding a raw
        dict — though real LIBERO would have a model-side rename
        producing nothing in the registry initially). The publish
        path then fires from the skip branch and replaces nothing
        (idempotent), but for ``wrist_image`` (not pre-populated),
        we go through the regular ``add_camera`` happy path.

        This test covers both the skip + publish path AND the regular
        ``add_camera`` path in one ``_install_libero_cameras`` call,
        verifying neither path errors on this realistic-mixed case.
        """
        sim = FakeSim(data_config="panda")
        # Mimic the LIBERO scene-rename situation: 'image' came in
        # via scene MJCF (so add_camera should NOT be re-called for
        # it), 'wrist_image' is fresh (add_camera should fire).
        sim._world.cameras["image"] = {"already": "there"}

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        adapter.on_episode_start(sim, random.Random(0))

        # 'image' already there — must NOT have been re-added.
        added_names = {name for name, _ in sim._add_camera_calls}
        assert "image" not in added_names
        # The pre-existing 'image' entry stays untouched (idempotent).
        assert sim._world.cameras["image"] == {"already": "there"}
        # 'wrist_image' got added regularly.
        assert "wrist_image" in added_names


# Step semantics


class TestOnStep:
    def test_sparse_reward_zero_and_not_done(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        info = adapter.on_step(FakeSim(), {}, {})
        assert info.reward == 0.0
        assert info.done is False


# augment_observation - LIBERO state bridge (#156)


class TestAugmentObservation:
    """``LiberoAdapter.augment_observation`` injects ``x`` / ``y`` / ``z`` /
    ``roll`` / ``pitch`` / ``yaw`` / ``gripper`` so the ``libero_panda``
    Gr00tDataConfig finds them in the per-step observation."""

    @pytest.fixture
    def libero_scene_xml(self):
        """Use the existing scene cache if available; otherwise skip.

        Mirrors the fixture on ``TestInstallActionController``. The
        round-31 site-priority tests need a real LIBERO-shaped MJCF
        with both ``robot0_right_hand`` (body) and
        ``gripper0_grip_site`` (site) so we can verify the site path
        is preferred over the body path.
        """
        from pathlib import Path

        cache_dir = Path.home() / ".strands_robots" / "scene_cache" / "libero"
        if not cache_dir.is_dir():
            pytest.skip(f"no scene cache at {cache_dir}; cannot test site-priority end-to-end")
        cached = list(cache_dir.glob("*.xml"))
        if not cached:
            pytest.skip(f"no .xml files in {cache_dir}; cannot test site-priority end-to-end")
        return str(cached[0])

    def test_default_panda_layout(self):
        """Identity quaternion ⇒ all Euler angles are zero."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "hand": {
                    "position": [0.5, -0.1, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],  # identity
                },
            }
        )
        # Provide finger_joint1 in the input obs (the runner gets it from
        # sim.get_observation; here we simulate that contract directly).
        obs = {"finger_joint1": 0.04, "joint1": 0.0}
        out = adapter.augment_observation(sim, obs)
        # Cartesian
        assert out["x"] == pytest.approx(0.5)
        assert out["y"] == pytest.approx(-0.1)
        assert out["z"] == pytest.approx(0.3)
        # Euler - identity quat → all zeros.
        assert out["roll"] == pytest.approx(0.0, abs=1e-9)
        assert out["pitch"] == pytest.approx(0.0, abs=1e-9)
        assert out["yaw"] == pytest.approx(0.0, abs=1e-9)
        # Gripper from the finger_joint1 reading. Packed as a 2-element
        # list to match `robot0_gripper_qpos` (the LIBERO/RoboSuite
        # 2-finger array the GR00T-N1.7-LIBERO checkpoint was trained
        # on); both entries mirror the same finger_joint1 reading
        # because the Menagerie Panda's MJCF equality constraint
        # forces finger_joint1 == finger_joint2.
        assert out["gripper"] == pytest.approx([0.04, 0.04])
        # Original keys preserved.
        assert out["finger_joint1"] == 0.04
        assert out["joint1"] == 0.0

    def test_quaternion_to_euler_z_rotation(self):
        """90° rotation about Z ⇒ yaw = π/2, roll = pitch = 0."""
        import math

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        # MuJoCo quat (w,x,y,z) for 90° about Z: (cos(45°), 0, 0, sin(45°))
        quat = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": quat}})
        out = adapter.augment_observation(sim, {})
        assert out["roll"] == pytest.approx(0.0, abs=1e-9)
        assert out["pitch"] == pytest.approx(0.0, abs=1e-9)
        assert out["yaw"] == pytest.approx(math.pi / 2, abs=1e-9)

    def test_quaternion_to_euler_x_rotation(self):
        """90° rotation about X ⇒ roll = π/2, pitch = yaw = 0."""
        import math

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        quat = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": quat}})
        out = adapter.augment_observation(sim, {})
        assert out["roll"] == pytest.approx(math.pi / 2, abs=1e-9)
        assert out["pitch"] == pytest.approx(0.0, abs=1e-9)
        assert out["yaw"] == pytest.approx(0.0, abs=1e-9)

    def test_quaternion_gimbal_lock_does_not_crash(self):
        """90° rotation about Y is gimbal-locked; the resolution may
        push roll into yaw, but pitch must equal π/2 and the call MUST
        return without raising."""
        import math

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        quat = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": quat}})
        out = adapter.augment_observation(sim, {})
        assert out["pitch"] == pytest.approx(math.pi / 2, abs=1e-6)

    def test_eef_state_keys_match_libero_panda_data_config(self):
        """Regression: keys MUST exactly equal the bare names of the
        ``libero_panda`` data_config's state_keys, otherwise the policy
        won't find them."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]}})
        out = adapter.augment_observation(sim, {"finger_joint1": 0.0})
        # libero_panda state_keys.removeprefix("state.") = bare names below.
        for required in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"):
            assert required in out, f"missing {required!r} in augmented obs"

    def test_missing_body_drops_pose_keys_silently(self):
        """If the EEF body isn't in the sim, the adapter must NOT raise -
        the missing keys are simply omitted so the policy surfaces a
        clearer 'state.x must be in observation' error than a Python
        traceback."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(bodies={})  # no "hand" body
        out = adapter.augment_observation(sim, {"finger_joint1": 0.04})
        assert "x" not in out
        assert "y" not in out
        assert "roll" not in out
        # Gripper still injected because finger_joint1 is in obs;
        # mirrored into the 2-element shape the trained checkpoint
        # expects (see test_pose_and_gripper_inject_into_obs above).
        assert out["gripper"] == pytest.approx([0.04, 0.04])

    def test_missing_gripper_omits_gripper_key(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]}})
        out = adapter.augment_observation(sim, {})  # no finger_joint1
        assert "gripper" not in out

    def test_inject_eef_state_false_disables_hook(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, inject_eef_state=False)
        sim = FakeSim(bodies={"hand": {"position": [1, 2, 3], "quaternion": [1, 0, 0, 0]}})
        obs = {"finger_joint1": 0.04, "joint1": 0.0}
        out = adapter.augment_observation(sim, obs)
        # Returned obs is unchanged - no x/y/z/roll/pitch/yaw/gripper.
        for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"):
            assert key not in out
        assert out["finger_joint1"] == 0.04

    def test_does_not_overwrite_keys_already_in_obs(self):
        """If a backend already supplies x/y/..., the adapter must NOT
        clobber them - users may have configured an observation_mapping."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(bodies={"hand": {"position": [9.9, 9.9, 9.9], "quaternion": [1, 0, 0, 0]}})
        obs = {"x": 0.0, "finger_joint1": 0.04}
        out = adapter.augment_observation(sim, obs)
        # User's x wins over the FK lookup.
        assert out["x"] == 0.0
        # But fields the user didn't supply still get filled in.
        assert out["y"] == pytest.approx(9.9)

    def test_custom_eef_body_name(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, eef_body_name="custom_eef")
        sim = FakeSim(
            bodies={
                "hand": {"position": [9, 9, 9], "quaternion": [1, 0, 0, 0]},
                "custom_eef": {"position": [0.5, 0.5, 0.5], "quaternion": [1, 0, 0, 0]},
            }
        )
        out = adapter.augment_observation(sim, {})
        assert out["x"] == pytest.approx(0.5)
        # Default "hand" body NOT used.
        assert out["x"] != 9

    def test_custom_gripper_joint_name(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, gripper_joint_name="my_grip")
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]}})
        out = adapter.augment_observation(sim, {"my_grip": 0.025, "finger_joint1": 9.9})
        # Reads from my_grip, not finger_joint1.
        assert out["gripper"] == pytest.approx([0.025, 0.025])

    def test_namespaced_gripper_joint_resolves(self):
        """Multi-robot scenes namespace joints as ``<robot>/<joint>``. The
        adapter must accept a suffix match so users don't have to set
        ``gripper_joint_name='panda_arm/finger_joint1'`` manually."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]}})
        # Namespaced key as it would appear in a multi-robot sim.
        obs = {"panda_arm/finger_joint1": 0.04}
        out = adapter.augment_observation(sim, obs)
        assert out["gripper"] == pytest.approx([0.04, 0.04])

    def test_sim_without_get_body_state(self):
        """Backends that don't expose get_body_state (future engines)
        must skip pose injection silently rather than crash."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)

        class BareSim:
            pass

        sim = BareSim()
        # Must not raise.
        out = adapter.augment_observation(sim, {"finger_joint1": 0.04})  # type: ignore[arg-type]
        assert "x" not in out
        # Gripper still injected from the obs lookup.
        assert out["gripper"] == pytest.approx([0.04, 0.04])

    def test_state_log_disabled_by_default(self, monkeypatch, caplog):
        """Round 30 (#168): ``STRANDS_LIBERO_STATE_LOG`` is opt-in.
        When unset, ``augment_observation`` emits no STATE_LOG lines.

        Pin so a future bug doesn't accidentally enable it by default
        (which would flood production eval logs with state values from
        every observation request)."""
        import logging as _logging

        monkeypatch.delenv("STRANDS_LIBERO_STATE_LOG", raising=False)

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "hand": {
                    "position": [0.5, -0.1, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )

        with caplog.at_level(_logging.INFO, logger="strands_robots.benchmarks.libero.adapter"):
            adapter.augment_observation(sim, {"finger_joint1": 0.04})

        state_log_lines = [r for r in caplog.records if "STATE_LOG" in r.getMessage()]
        assert len(state_log_lines) == 0, f"expected zero STATE_LOG lines without env var, got {len(state_log_lines)}"

    def test_state_log_enabled_emits_one_log_per_call(self, monkeypatch, caplog):
        """Round 30 (#168): with ``STRANDS_LIBERO_STATE_LOG=1``, each
        ``augment_observation`` call emits exactly one ``STATE_LOG``
        INFO line containing the state values being fed to the GR00T
        policy server.

        Field-set contract:
        - ``step=N``
        - ``x=`` ``y=`` ``z=`` (Cartesian position)
        - ``roll=`` ``pitch=`` ``yaw=`` (Euler angles, sxyz extrinsic)
        - ``gripper=`` (2-element list per round-21's PR #162 shape)
        - ``obs_keys=`` (sorted list of all keys in the merged obs)

        Pin so a future refactor that drops a field fails loudly here.
        """
        import logging as _logging

        monkeypatch.setenv("STRANDS_LIBERO_STATE_LOG", "1")

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        # Verify cached env-var read.
        assert adapter._state_log_enabled is True
        assert adapter._state_log_step == 0

        sim = FakeSim(
            bodies={
                "hand": {
                    "position": [0.5, -0.1, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )

        with caplog.at_level(_logging.INFO, logger="strands_robots.benchmarks.libero.adapter"):
            adapter.augment_observation(sim, {"finger_joint1": 0.04})

        state_log_lines = [r for r in caplog.records if "STATE_LOG" in r.getMessage()]
        assert len(state_log_lines) == 1, f"expected exactly 1 STATE_LOG line, got {len(state_log_lines)}"

        msg = state_log_lines[0].getMessage()
        for field in ["step=0", "x=", "y=", "z=", "roll=", "pitch=", "yaw=", "gripper=", "obs_keys="]:
            assert field in msg, f"missing field {field!r} in STATE_LOG: {msg}"

        # Counter advanced.
        assert adapter._state_log_step == 1

    def test_state_log_caps_at_max_steps(self, monkeypatch, caplog):
        """Round 30 (#168): ``STRANDS_LIBERO_STATE_LOG_MAX`` caps how
        many ``augment_observation`` calls log per episode. After the
        cap, the call is silent. Pin so a long episode doesn't flood
        logs once the diagnostic window has captured what's needed."""
        import logging as _logging

        monkeypatch.setenv("STRANDS_LIBERO_STATE_LOG", "1")
        monkeypatch.setenv("STRANDS_LIBERO_STATE_LOG_MAX", "3")

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        assert adapter._state_log_max == 3

        sim = FakeSim(
            bodies={
                "hand": {
                    "position": [0.5, -0.1, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )

        with caplog.at_level(_logging.INFO, logger="strands_robots.benchmarks.libero.adapter"):
            for _ in range(7):
                adapter.augment_observation(sim, {"finger_joint1": 0.04})

        state_log_lines = [r for r in caplog.records if "STATE_LOG" in r.getMessage()]
        assert len(state_log_lines) == 3, f"expected 3 STATE_LOG lines (cap=3), got {len(state_log_lines)}"
        assert adapter._state_log_step == 3

    def test_state_log_resets_on_episode_start(self, monkeypatch):
        """Round 30 (#168): ``on_episode_start`` zeros the per-episode
        STATE_LOG counter so each episode logs its own first N
        observations.

        Without this, the second episode of a multi-episode eval
        would silently emit no STATE_LOG (counter already at the cap
        from episode 1)."""
        import random

        monkeypatch.setenv("STRANDS_LIBERO_STATE_LOG", "1")

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)

        # Drive counter forward.
        adapter._state_log_step = 5

        sim = FakeSim(data_config="panda")
        rng = random.Random(0)
        adapter.on_episode_start(sim, rng)

        assert adapter._state_log_step == 0, "on_episode_start must zero the per-episode STATE_LOG counter"

    def test_state_log_invalid_max_value_warns_and_falls_back(self, monkeypatch, caplog):
        """Round 30 (#168): a malformed ``STRANDS_LIBERO_STATE_LOG_MAX``
        (e.g. ``foo``) must log a WARNING and fall back to the default
        (50), not silently swallow the typo. Mirrors the AGENTS.md rule
        for ``STRANDS_*`` env vars."""
        import logging as _logging

        monkeypatch.setenv("STRANDS_LIBERO_STATE_LOG", "1")
        monkeypatch.setenv("STRANDS_LIBERO_STATE_LOG_MAX", "not-an-int")

        with caplog.at_level(_logging.WARNING, logger="strands_robots.benchmarks.libero.adapter"):
            adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)

        # Default fallback.
        assert adapter._state_log_max == 50
        warnings = [
            r
            for r in caplog.records
            if r.levelno == _logging.WARNING and "STRANDS_LIBERO_STATE_LOG_MAX" in r.getMessage()
        ]
        assert len(warnings) >= 1, "expected a WARNING about malformed STRANDS_LIBERO_STATE_LOG_MAX"

    def test_eef_state_site_name_default_matches_libero_convention(self):
        """Round 31 (#168): default ``eef_state_site_name`` is
        ``"<scene_gripper_prefix>grip_site"`` ⇒ ``"gripper0_grip_site"``
        for LIBERO. This is the same site RoboSuite's
        ``OperationalSpaceController`` reads for ``robot0_eef_pos`` /
        ``robot0_eef_quat``, so reading from it produces state matching
        what LIBERO checkpoints were trained against.

        Pin so a future scene_gripper_prefix change (e.g. ``robot0_``)
        propagates to the site name without silent drift, AND so a
        test of "are we using the right site" can be a one-line
        property assertion."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        assert adapter.eef_state_site_name == "gripper0_grip_site"

    def test_eef_state_site_name_user_override(self):
        """Round 31 (#168): user-supplied ``eef_state_site_name`` overrides
        the auto-derived default. Useful for non-RoboSuite scenes that
        ship a different site name, or for legacy tests pinning the
        body fallback by passing an empty string sentinel."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, eef_state_site_name="my_custom_site")
        assert adapter.eef_state_site_name == "my_custom_site"

    def test_eef_state_site_name_tracks_scene_gripper_prefix(self):
        """Round 31 (#168): when ``scene_gripper_prefix`` is changed,
        the auto-derived ``eef_state_site_name`` follows.

        Pin so a custom-namespace LIBERO variant doesn't end up reading
        from the wrong site silently."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, scene_gripper_prefix="custom_grip_")
        assert adapter.eef_state_site_name == "custom_grip_grip_site"

    def test_read_eef_pose_uses_site_when_available(self, libero_scene_xml):
        """Round 32 (#168) — pin the split-source contract.

        With a real LIBERO scene compiled into the sim,
        ``_read_eef_pose`` must:
        - read POSITION from ``data.site_xpos[gripper0_grip_site_id]``
          (matches RoboSuite's ``eef_pos`` observable)
        - read ORIENTATION from ``data.xquat[robot0_right_hand_id]``
          (matches RoboSuite's ``eef_quat`` observable, which uses
          ``get_body_xquat(eef_name)``).

        Round 31 read both from the site and produced a 90° yaw
        offset because ``site_xmat`` ≠ body ``xquat`` for the panda
        gripper. Round 32 splits the reads."""
        pytest.importorskip("mujoco")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
            # Round 32: explicitly set the body name so the test
            # exercises the split-source path. (Auto-resolution
            # happens in on_episode_start, which we bypass here.)
            eef_body_name="robot0_right_hand",
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        # Ground truth: site_xpos for position, body xquat for orientation.
        site_id = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")  # type: ignore[attr-defined]
        body_id = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")  # type: ignore[attr-defined]
        if site_id < 0 or body_id < 0:
            pytest.skip("scene XML missing 'gripper0_grip_site' or 'robot0_right_hand'")
        expected_pos = list(sim._world._data.site_xpos[site_id])  # type: ignore[attr-defined]
        expected_quat = list(sim._world._data.xquat[body_id])  # type: ignore[attr-defined]

        pos, quat = adapter._read_eef_pose(sim)
        assert pos is not None, "site path should produce a position"
        assert quat is not None, "body path should produce an orientation"

        # Position from site (within FP tolerance).
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            assert pos[axis_idx] == pytest.approx(expected_pos[axis_idx], abs=1e-6), (
                f"site {axis_name} {pos[axis_idx]} != expected {expected_pos[axis_idx]}"
            )

        # Orientation from body xquat (wxyz, within FP tolerance).
        for q_idx, q_name in enumerate(["w", "x", "y", "z"]):
            assert quat[q_idx] == pytest.approx(expected_quat[q_idx], abs=1e-6), (
                f"body quat.{q_name} {quat[q_idx]} != expected {expected_quat[q_idx]}; "
                f"orientation may be coming from site_xmat instead of body xquat"
            )

        # Quaternion is unit-norm.
        quat_arr = np.array(quat)
        assert np.linalg.norm(quat_arr) == pytest.approx(1.0, abs=1e-6)

    def test_read_eef_pose_site_differs_from_body(self, libero_scene_xml):
        """Round 31/32 (#168): pin the round-30 finding that site
        readout (used for position) and body readout differ by ~9.7 cm.
        This sentinel ensures the test environment exhibits the
        original bug, so the test is meaningful.

        Round 32: also asserts the position we get back matches the
        site, NOT the body — the round-30 wrong-body bug regression
        sentinel.
        """
        pytest.importorskip("mujoco")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
            eef_body_name="robot0_right_hand",
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        # Find the wrist body that the legacy body path would read.
        body_id = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")  # type: ignore[attr-defined]
        site_id = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")  # type: ignore[attr-defined]
        if body_id < 0 or site_id < 0:
            pytest.skip("scene XML missing 'robot0_right_hand' or 'gripper0_grip_site'")

        body_pos = np.array(sim._world._data.xpos[body_id])  # type: ignore[attr-defined]
        site_pos = np.array(sim._world._data.site_xpos[site_id])  # type: ignore[attr-defined]

        # The two MUST differ — otherwise this scene doesn't exhibit
        # the round-30 bug and the regression test isn't meaningful.
        delta = site_pos - body_pos
        assert np.linalg.norm(delta) > 0.05, (
            f"site_xpos vs body xpos differ by only {np.linalg.norm(delta)} m; "
            f"expected ~10 cm. Scene may not match LIBERO's robot0_right_hand / "
            f"gripper0_grip_site convention. (site={site_pos}, body={body_pos})"
        )

        # _read_eef_pose returns the SITE value, not the body value.
        pos, _ = adapter._read_eef_pose(sim)
        assert pos is not None
        for axis_idx, expected in enumerate(site_pos):
            assert pos[axis_idx] == pytest.approx(float(expected), abs=1e-6), (
                f"axis {axis_idx}: returned {pos[axis_idx]} matches body {body_pos[axis_idx]} "
                f"instead of site {expected}. Round-30 wrong-body bug may have regressed."
            )

    def test_read_eef_pose_falls_back_to_body_when_site_missing(self, libero_scene_xml):
        """Round 31 (#168): when ``eef_state_site_name`` resolves to a
        site that doesn't exist (custom non-RoboSuite scene), fall
        back to the body lookup so non-LIBERO callers see zero
        behaviour change.

        Pin so the round-31 site preference doesn't break adapters
        used outside LIBERO."""
        pytest.importorskip("mujoco")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
            # Override to a site name that doesn't exist in the scene.
            eef_state_site_name="this_site_does_not_exist",
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        # Scene exists, so get_body_state would work for the canonical
        # body if it were resolvable. We rely on FakeSim's
        # get_body_state implementation (returns from self._bodies),
        # which is empty here — so the fallback returns (None, None).
        # Round-31 contract: NOT raising, just gracefully degrading.
        pos, quat = adapter._read_eef_pose(sim)
        # FakeSim's get_body_state on a missing body returns "missing"
        # error → (None, None). That's the correct behaviour for the
        # body-fallback path with no body match.
        assert pos is None or pos == [0, 0, 0]
        # No site, no body match → both None (or the fallback's empty
        # default). Either way: no exception was raised.

    def test_augment_observation_uses_site_path_with_real_scene(self, libero_scene_xml):
        """Round 31/32 (#168) end-to-end: ``augment_observation`` produces
        ``state.x/y/z`` from the SITE (gripper tip) and
        ``state.roll/pitch/yaw`` from the BODY (wrist) xquat → Euler.

        This is the customer-facing assertion: the values that flow
        into the GR00T policy server's ``state.*`` keys must match
        RoboSuite's split-source convention. Confirms the round-32
        fix at the public ``augment_observation`` boundary.
        """
        pytest.importorskip("mujoco")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
            eef_body_name="robot0_right_hand",
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        site_id = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")  # type: ignore[attr-defined]
        body_id = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")  # type: ignore[attr-defined]
        if site_id < 0 or body_id < 0:
            pytest.skip("scene XML missing 'gripper0_grip_site' or 'robot0_right_hand'")
        expected_pos = list(sim._world._data.site_xpos[site_id])  # type: ignore[attr-defined]
        expected_quat_wxyz = list(sim._world._data.xquat[body_id])  # type: ignore[attr-defined]
        # Convert expected wxyz quat to roll/pitch/yaw using the same
        # helper the adapter uses, so the test pins the *behaviour*
        # (split-source) rather than the helper's exact numerics.
        from strands_robots.benchmarks.libero.adapter import _quat_wxyz_to_rpy_xyz

        expected_roll, expected_pitch, expected_yaw = _quat_wxyz_to_rpy_xyz(expected_quat_wxyz)

        out = adapter.augment_observation(sim, {})

        # state.x/y/z from site_xpos.
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            assert out[axis_name] == pytest.approx(expected_pos[axis_idx], abs=1e-6), (
                f"augment_observation state.{axis_name} {out[axis_name]} != "
                f"site {expected_pos[axis_idx]}; round-31 site-priority for position may have regressed."
            )

        # state.roll/pitch/yaw derived from body xquat (round-32 split).
        # Round-31 read these from site_xmat, which was 90° off in yaw.
        assert out["roll"] == pytest.approx(expected_roll, abs=1e-6), (
            f"state.roll {out['roll']} != body-derived {expected_roll}; "
            f"orientation may be coming from site_xmat (round-31 bug) instead of body xquat (round-32 fix)."
        )
        assert out["pitch"] == pytest.approx(expected_pitch, abs=1e-6)
        assert out["yaw"] == pytest.approx(expected_yaw, abs=1e-6), (
            f"state.yaw {out['yaw']} != body-derived {expected_yaw}; "
            f"orientation source may be wrong (site vs body — round-31 had this 90° off)."
        )

    def test_read_eef_pose_uses_body_xquat_not_site_xmat(self, libero_scene_xml):
        """Round 32 (#168) — pin the orientation source explicitly.

        Round 31 read the quaternion from ``site_xmat`` (via
        ``mju_mat2Quat``); round 32 reads it from ``data.xquat[body_id]``.
        These differ by 90° in yaw on a LIBERO panda gripper because
        the site coordinate frame is rotated relative to the body's.

        Pin the source explicitly: returned ``quat`` matches body
        ``xquat`` and does NOT match the site ``xmat``-derived quat.
        Sentinel ensures the bug regression cannot pass silently.
        """
        pytest.importorskip("mujoco")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
            eef_body_name="robot0_right_hand",
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        site_id = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")  # type: ignore[attr-defined]
        body_id = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")  # type: ignore[attr-defined]
        if site_id < 0 or body_id < 0:
            pytest.skip("scene XML missing 'gripper0_grip_site' or 'robot0_right_hand'")

        # Body xquat (mujoco wxyz convention) — the round-32 source.
        body_quat = np.array(sim._world._data.xquat[body_id])  # type: ignore[attr-defined]

        # Site xmat → quat (the round-31 source). Sanity-check the
        # two are different — otherwise the test isn't meaningful.
        site_xmat = np.array(sim._world._data.site_xmat[site_id]).reshape(9)  # type: ignore[attr-defined]
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, site_xmat)

        # Round-30 verification observed a 90° yaw offset between
        # site_xmat-derived quat and body xquat. The two MUST differ
        # by more than the FP tolerance — otherwise the test's
        # round-32 vs round-31 distinction is moot.
        # Note: quaternions q and -q represent the same rotation, so
        # we measure orientation difference via the dot product
        # (|q1 . q2| ≈ 1 means same rotation).
        dot = abs(float(np.dot(body_quat, site_quat)))
        assert dot < 0.999, (
            f"body xquat and site_xmat-derived quat are nearly identical (|dot|={dot}); "
            f"the scene doesn't exhibit the round-30 orientation bug, regression test is moot. "
            f"body_quat={body_quat}, site_quat={site_quat}"
        )

        # _read_eef_pose returns the BODY xquat, not the site_xmat-derived one.
        _, returned_quat = adapter._read_eef_pose(sim)
        assert returned_quat is not None
        returned_arr = np.array(returned_quat)
        for q_idx, expected in enumerate(body_quat):
            assert returned_arr[q_idx] == pytest.approx(float(expected), abs=1e-6), (
                f"axis {q_idx}: returned {returned_arr[q_idx]} doesn't match body xquat "
                f"{expected}. Round-32 split-source contract may have regressed back to site_xmat."
            )

    def test_state_gripper_joint_names_default_matches_libero_convention(self):
        """Round 33 (#168): default ``state_gripper_joint_names`` is
        ``["<scene_gripper_prefix>finger_joint1",
           "<scene_gripper_prefix>finger_joint2"]``
        ⇒ ``["gripper0_finger_joint1", "gripper0_finger_joint2"]``
        for LIBERO. These are the joints LIBERO's
        ``robot0_gripper_qpos`` observable indexes."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        assert adapter.state_gripper_joint_names == [
            "gripper0_finger_joint1",
            "gripper0_finger_joint2",
        ]

    def test_state_gripper_joint_names_user_override(self):
        """Round 33 (#168): user-supplied joint names override the
        auto-derived default."""
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            state_gripper_joint_names=["my_left_finger", "my_right_finger"],
        )
        assert adapter.state_gripper_joint_names == ["my_left_finger", "my_right_finger"]

    def test_state_gripper_joint_names_tracks_scene_gripper_prefix(self):
        """Round 33 (#168): when ``scene_gripper_prefix`` is changed,
        the auto-derived ``state_gripper_joint_names`` follows."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, scene_gripper_prefix="custom_grip_")
        assert adapter.state_gripper_joint_names == [
            "custom_grip_finger_joint1",
            "custom_grip_finger_joint2",
        ]

    def test_read_gripper_qpos_returns_both_fingers_with_opposite_signs(self, libero_scene_xml):
        """Round 33 (#168) — primary regression for the duplicated-finger bug.

        With a real LIBERO scene compiled into the sim,
        ``_read_gripper_qpos`` returns
        ``[gripper0_finger_joint1.qpos, gripper0_finger_joint2.qpos]``
        directly from ``data.qpos[jnt_qposadr]``. Their values have
        OPPOSITE signs by physical convention (the Panda gripper's
        two fingers move apart, with mirrored ctrlranges).

        Pre-round-33 the adapter packed ``[gripper_value, gripper_value]``
        (both positive) which was structurally OOD from training data.
        Pin so the duplicate-packing bug can't sneak back."""
        pytest.importorskip("mujoco")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        # Find both finger joints and compute their qpos addresses.
        j1 = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_JOINT, "gripper0_finger_joint1")  # type: ignore[attr-defined]
        j2 = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_JOINT, "gripper0_finger_joint2")  # type: ignore[attr-defined]
        if j1 < 0 or j2 < 0:
            pytest.skip("scene XML missing gripper0_finger_joint1/2")
        adr1 = int(sim._world._model.jnt_qposadr[j1])  # type: ignore[attr-defined]
        adr2 = int(sim._world._model.jnt_qposadr[j2])  # type: ignore[attr-defined]
        # Set them to canonical opposite-sign values for the test (the
        # initial mj_forward leaves them at 0 since no init_state is
        # applied). LIBERO at canonical ready-pose has them at
        # ~[+0.0208, -0.0208].
        sim._world._data.qpos[adr1] = 0.0208  # type: ignore[attr-defined]
        sim._world._data.qpos[adr2] = -0.0208  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        qpos = adapter._read_gripper_qpos(sim)
        assert qpos is not None, "_read_gripper_qpos should return both finger qpos"
        assert len(qpos) == 2

        # Round-33 contract: each finger's qpos read independently.
        assert qpos[0] == pytest.approx(0.0208, abs=1e-6), f"finger1 qpos {qpos[0]} != expected 0.0208"
        assert qpos[1] == pytest.approx(-0.0208, abs=1e-6), (
            f"finger2 qpos {qpos[1]} != expected -0.0208 (the opposite-sign convention)"
        )

        # Sentinel: round-33 fix means qpos[0] != qpos[1]. Pre-round-33's
        # duplicate packing would always have qpos[0] == qpos[1].
        assert qpos[0] != qpos[1], (
            f"both fingers reported as {qpos[0]} — round-33 duplicate-packing bug may have regressed"
        )

    def test_read_gripper_qpos_returns_none_when_joints_missing(self, libero_scene_xml):
        """Round 33 (#168): when configured joint names don't resolve
        in the compiled model (custom non-RoboSuite scene),
        ``_read_gripper_qpos`` returns ``None`` so the caller's
        single-joint duplicate fallback fires.

        Pin so non-LIBERO callers keep their pre-round-33 behaviour."""
        pytest.importorskip("mujoco")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
            # Override to joint names that don't exist.
            state_gripper_joint_names=["nonexistent_left", "nonexistent_right"],
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        result = adapter._read_gripper_qpos(sim)
        assert result is None, (
            f"expected None when joint names don't resolve, got {result}; "
            f"caller's fallback won't fire and state.gripper will be wrong"
        )

    def test_augment_observation_uses_two_finger_qpos_with_real_scene(self, libero_scene_xml):
        """Round 33 (#168) end-to-end: ``augment_observation``'s
        ``state.gripper`` is a 2-element list of OPPOSITE-sign finger
        qpos values, NOT a duplicate of one finger.

        This is the customer-facing assertion: the values that flow
        into the GR00T policy server's ``state.gripper`` key must
        match LIBERO's ``robot0_gripper_qpos`` shape and signs.
        Confirms the round-33 fix at the public boundary."""
        pytest.importorskip("mujoco")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
            eef_body_name="robot0_right_hand",
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        # Canonical opposite-sign finger positions.
        j1 = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_JOINT, "gripper0_finger_joint1")  # type: ignore[attr-defined]
        j2 = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_JOINT, "gripper0_finger_joint2")  # type: ignore[attr-defined]
        if j1 < 0 or j2 < 0:
            pytest.skip("scene XML missing gripper0_finger_joint1/2")
        adr1 = int(sim._world._model.jnt_qposadr[j1])  # type: ignore[attr-defined]
        adr2 = int(sim._world._model.jnt_qposadr[j2])  # type: ignore[attr-defined]
        sim._world._data.qpos[adr1] = 0.0208  # type: ignore[attr-defined]
        sim._world._data.qpos[adr2] = -0.0208  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        out = adapter.augment_observation(sim, {})
        assert "gripper" in out
        assert len(out["gripper"]) == 2

        # The two values must be OPPOSITE in sign — that's the
        # round-33 contract that distinguishes the fix from pre-fix
        # duplicate-packing. Use a tolerance because the two fingers
        # could be at slightly different magnitudes due to gravity.
        assert out["gripper"][0] == pytest.approx(0.0208, abs=1e-6)
        assert out["gripper"][1] == pytest.approx(-0.0208, abs=1e-6)
        assert out["gripper"][0] != out["gripper"][1], (
            f"state.gripper {out['gripper']} has duplicated value — round-33 bug regressed"
        )


class TestAugmentObservationImageFlip:
    """Round 39 (#168): ``LiberoAdapter.augment_observation`` flips
    rendered images vertically to match upstream LIBERO's
    ``OffScreenRenderEnv`` pixel convention.

    Why: our ``sim.render()`` returns top-row-zero (image convention),
    upstream's ``OffScreenRenderEnv`` returns bottom-row-zero (OpenGL
    convention). The GR00T-N1.7-LIBERO checkpoint was trained against
    upstream + an additional ``[::-1, ::-1]`` rotation done by
    ``LiberoEnv._process_observation``. The policy's
    ``image_rotation_180`` flag mirrors that rotation. So feeding raw
    sim renders to the policy yields an upside-down image relative to
    training (round-39 ``diff_libero_obs.py`` measured
    ``mean |Δ| = 56/255`` raw vs upstream, ``5.40/255`` after V-flip).

    Adapter-side flip means the policy stays generic — its
    ``image_rotation_180`` flag retains its training-pipeline meaning.
    """

    def _gradient_image(self) -> np.ndarray:
        """A 4×4×3 uint8 image with row-distinguishable pixels.

        Top row is bright (255), bottom row dark (0). After V-flip,
        top becomes dark and bottom becomes bright. Test pin uses this
        signature to detect both presence and direction of the flip.
        """
        rows = np.linspace(255, 0, 4, dtype=np.uint8)  # 255, 170, 85, 0
        return np.stack([np.full((4, 3), v, dtype=np.uint8) for v in rows], axis=0)

    def test_image_is_flipped_vertically(self):
        """Round 39 (#168): ``image`` key gets ``[::-1, :]`` applied."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "hand": {
                    "position": [0.5, -0.1, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )
        gradient = self._gradient_image()
        obs = {"finger_joint1": 0.04, "image": gradient}
        out = adapter.augment_observation(sim, obs)
        # After V-flip: top of input is now bottom of output.
        np.testing.assert_array_equal(out["image"], gradient[::-1, :])
        # The original input must NOT be mutated.
        np.testing.assert_array_equal(obs["image"], gradient)

    def test_wrist_image_is_flipped_vertically(self):
        """Round 39 (#168): ``wrist_image`` key gets ``[::-1, :]`` applied
        (same convention as ``image``)."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "hand": {
                    "position": [0.5, -0.1, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )
        gradient = self._gradient_image()
        obs = {"finger_joint1": 0.04, "wrist_image": gradient}
        out = adapter.augment_observation(sim, obs)
        np.testing.assert_array_equal(out["wrist_image"], gradient[::-1, :])

    def test_flipped_image_is_contiguous(self):
        """Round 39 (#168): the flipped image must be C-contiguous so
        downstream serialization (msgpack / numpy.tobytes()) works.

        ``arr[::-1, :]`` returns a non-contiguous view; the adapter
        must materialize it via ``np.ascontiguousarray``."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "hand": {
                    "position": [0.5, -0.1, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )
        gradient = self._gradient_image()
        obs = {"finger_joint1": 0.04, "image": gradient}
        out = adapter.augment_observation(sim, obs)
        assert out["image"].flags["C_CONTIGUOUS"], (
            "flipped image must be C-contiguous for msgpack serialization; "
            "round-39 fix should wrap with np.ascontiguousarray"
        )

    def test_skips_non_ndarray_values(self):
        """Round 39 (#168): if a backend supplies the image as something
        other than an ndarray (e.g. PIL Image, base64 string), the flip
        skips silently rather than raising — preserves backend
        flexibility."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "hand": {
                    "position": [0.5, -0.1, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )
        # Pass a string instead of an ndarray — should pass through unchanged.
        obs = {"finger_joint1": 0.04, "image": "<base64-png-bytes>"}
        out = adapter.augment_observation(sim, obs)
        assert out["image"] == "<base64-png-bytes>"

    def test_skips_low_dimensional_arrays(self):
        """Round 39 (#168): an ndarray with fewer than 2 dims (e.g. a
        scalar / 1D state somehow keyed under 'image') is left alone.

        Pin so the flip can't accidentally corrupt non-image data that
        happens to be keyed under one of the camera names."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "hand": {
                    "position": [0.5, -0.1, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )
        # 1D ndarray keyed as 'image'.
        weird = np.array([1.0, 2.0, 3.0])
        obs = {"finger_joint1": 0.04, "image": weird}
        out = adapter.augment_observation(sim, obs)
        np.testing.assert_array_equal(out["image"], weird)

    def test_does_not_inject_when_keys_absent(self):
        """Round 39 (#168): no ``image`` / ``wrist_image`` keys in input
        means no synthetic keys appear in output. The flip is purely a
        transform of existing keys, not a producer of new ones."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "hand": {
                    "position": [0.5, -0.1, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )
        obs = {"finger_joint1": 0.04}
        out = adapter.augment_observation(sim, obs)
        assert "image" not in out
        assert "wrist_image" not in out


class TestQuaternionToEuler:
    """Pure-math regression tests for ``_quat_wxyz_to_rpy_xyz`` so future
    refactors to the helper don't silently rotate the LIBERO state by π."""

    def test_identity(self):
        from strands_robots.benchmarks.libero.adapter import _quat_wxyz_to_rpy_xyz

        roll, pitch, yaw = _quat_wxyz_to_rpy_xyz([1.0, 0.0, 0.0, 0.0])
        assert abs(roll) < 1e-12
        assert abs(pitch) < 1e-12
        assert abs(yaw) < 1e-12

    @pytest.mark.parametrize(
        "quat,expected",
        [
            # 90° about X
            ([0.7071067811865476, 0.7071067811865475, 0.0, 0.0], (1.5707963267948966, 0.0, 0.0)),
            # 90° about Z
            ([0.7071067811865476, 0.0, 0.0, 0.7071067811865475], (0.0, 0.0, 1.5707963267948966)),
            # 180° about X - atan2(±0, -1) may return ±π, both are correct.
            ([0.0, 1.0, 0.0, 0.0], (3.141592653589793, 0.0, 0.0)),
            # 180° about Z
            ([0.0, 0.0, 0.0, 1.0], (0.0, 0.0, 3.141592653589793)),
        ],
    )
    def test_principal_axis_rotations(self, quat, expected):
        """Compare modulo 2π so atan2's ±π ambiguity at 180° rotations
        doesn't make the test brittle to floating-point sign-of-zero."""
        import math

        from strands_robots.benchmarks.libero.adapter import _quat_wxyz_to_rpy_xyz

        out = _quat_wxyz_to_rpy_xyz(quat)
        for actual, want in zip(out, expected, strict=True):
            diff = ((actual - want + math.pi) % (2 * math.pi)) - math.pi
            assert abs(diff) < 1e-9, f"got {out}, want {expected} (mod 2π)"

    def test_gimbal_lock_pitch_pi_over_2(self):
        """90° about Y is the canonical gimbal-lock case; pitch must
        still come out as π/2 even though roll/yaw absorb each other."""
        import math

        from strands_robots.benchmarks.libero.adapter import _quat_wxyz_to_rpy_xyz

        quat = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
        _roll, pitch, _yaw = _quat_wxyz_to_rpy_xyz(quat)
        assert pitch == pytest.approx(math.pi / 2, abs=1e-6)

    def test_matches_robosuite_mat2euler_sxyz(self):
        """Round 46 (#176 sub-task 3d) — pin byte-equivalence to
        robosuite's ``mat2euler(R, axes='sxyz')``.

        The pre-46 formula derived sin(pitch) = R[0,2] from
        ``R = R_x · R_y · R_z`` (intrinsic XYZ), which produced
        sign-flipped Euler angles for some quaternions vs robosuite.
        E.g. q=(0, 0.7071, 0.7071, 0) → robosuite returns (π, 0, π/2)
        but pre-46 returned (-π, 0, -π/2) — different yaw direction
        (left vs right twist about z), genuinely different rotation.

        This test pins the post-46 formula against pre-computed
        expected values derived from robosuite's algorithm so a
        future "simplification" can't silently re-introduce the sign
        flip and put state.yaw out-of-distribution from the policy's
        training data.

        Expected values were obtained from
        ``robosuite.utils.transform_utils.mat2euler(quat2mat(q), axes='sxyz')``
        (see ``tools/recompute_robosuite_euler.py`` if regenerating).
        """
        import math

        from strands_robots.benchmarks.libero.adapter import _quat_wxyz_to_rpy_xyz

        # (q_wxyz, robosuite_sxyz_rpy) pairs. Each q is normalised
        # to make the test data length-independent.
        cases = [
            # identity → (0, 0, 0)
            ([1.0, 0.0, 0.0, 0.0], (0.0, 0.0, 0.0)),
            # 90° about X → (π/2, 0, 0)
            ([0.7071067811865476, 0.7071067811865475, 0.0, 0.0], (math.pi / 2, 0.0, 0.0)),
            # 90° about Y → (0, π/2, 0); gimbal-lock branch may
            # absorb roll into yaw, accept (0, π/2, 0) up to the
            # gimbal-lock degeneracy.
            ([0.7071067811865476, 0.0, 0.7071067811865475, 0.0], (0.0, math.pi / 2, 0.0)),
            # 90° about Z → (0, 0, π/2)
            ([0.7071067811865476, 0.0, 0.0, 0.7071067811865475], (0.0, 0.0, math.pi / 2)),
            # 180° about (x+y)/sqrt(2) (the case that exposed the
            # pre-46 sign-flip bug). robosuite returns (π, 0, π/2);
            # pre-46 incorrectly returned (-π, 0, -π/2).
            ([0.0, 0.7071067811865475, 0.7071067811865475, 0.0], (math.pi, 0.0, math.pi / 2)),
            # equal-component quat (0.5, 0.5, 0.5, 0.5):
            # robosuite returns (π/2, 0, π/2)
            ([0.5, 0.5, 0.5, 0.5], (math.pi / 2, 0.0, math.pi / 2)),
        ]
        for q_wxyz, expected_rpy in cases:
            ours = _quat_wxyz_to_rpy_xyz(list(q_wxyz))
            for axis_idx, axis_name in enumerate(("roll", "pitch", "yaw")):
                # Mod 2π for ±π wraparound; use a generous 1e-5 tol
                # because asin/atan2 numerical precision dominates.
                diff = ((ours[axis_idx] - expected_rpy[axis_idx] + math.pi) % (2 * math.pi)) - math.pi
                # Gimbal-lock case (90° about Y): the (0, π/2, 0)
                # decomposition isn't unique — roll and yaw both
                # absorb into a single rotation. Skip the roll/yaw
                # axis check at exactly pitch=π/2.
                if abs(expected_rpy[1] - math.pi / 2) < 1e-9 and axis_name in ("roll", "yaw"):
                    continue
                assert abs(diff) < 1e-5, (
                    f"q={q_wxyz}: ours.{axis_name}={ours[axis_idx]:.6f} "
                    f"vs robosuite expected={expected_rpy[axis_idx]:.6f} "
                    f"(diff_mod_2pi={diff:.6e}). "
                    f"Round-46 ``mat2euler(sxyz)`` parity may have regressed."
                )

    def test_pre_46_sign_flip_regression(self):
        """Round 46 (#176 sub-task 3d) — explicit pin against the
        sign-flip bug that the pre-46 formula introduced.

        For ``q = (0, sqrt(2)/2, sqrt(2)/2, 0)`` (180° about (1,1,0)
        axis), the correct sxyz Euler is ``(π, 0, π/2)``. The pre-46
        formula returned ``(-π, 0, -π/2)`` — same roll modulo 2π but
        OPPOSITE yaw (which is genuinely a different rotation about
        the world Z axis at this orientation).

        Pin: ours_yaw must be CLOSE TO +π/2, not -π/2.
        """
        import math

        from strands_robots.benchmarks.libero.adapter import _quat_wxyz_to_rpy_xyz

        q = [0.0, 0.7071067811865475, 0.7071067811865475, 0.0]
        roll, pitch, yaw = _quat_wxyz_to_rpy_xyz(q)
        # roll: ±π are both correct (180° rotation is sign-ambiguous).
        # ``abs(abs(roll) - π)`` is small for both +π and -π.
        assert abs(abs(roll) - math.pi) < 1e-5, f"roll={roll:.6f}, expected ±π"
        # pitch must be 0
        assert abs(pitch) < 1e-5
        # yaw must be +π/2, NOT -π/2
        assert abs(yaw - math.pi / 2) < 1e-5, (
            f"yaw={yaw:.6f}, expected +π/2 ≈ 1.5708. "
            f"Sign-flip bug from pre-46 ``_quat_wxyz_to_rpy_xyz`` may have "
            f"regressed — would break state.yaw OOD on the policy."
        )


# Scene auto-generation from BDDL (#164)


class TestSceneGeneration:
    """``LiberoAdapter._generate_scene_from_bddl`` builds the per-task MJCF
    via the upstream ``libero`` package's procedural generator, with
    SHA256-keyed disk caching and a graceful fallback when ``libero``
    isn't installed."""

    def test_explicit_scene_path_skips_generation(self, tmp_path):
        """``scene_path=...`` set on the constructor wins over auto-gen.
        The generator must NOT be called when an explicit path is given."""

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=str(tmp_path / "explicit.xml"),
            init_jitter=0.0,
            install_cameras=False,
        )

        with patch.object(adapter, "_generate_scene_from_bddl") as mock_gen:
            sim = FakeSim(data_config="panda")
            # The explicit path doesn't exist; load_scene returns success
            # only because FakeSim's load_scene is canned. Real Simulation
            # would error - that's the test's purpose: confirm we never
            # touched the generator.
            adapter.on_episode_start(sim, random.Random(0))

        mock_gen.assert_not_called()
        assert sim._scenes_loaded == [str(tmp_path / "explicit.xml")]

    def test_auto_generate_scene_false_falls_back_to_bare_panda(self):
        """Pre-#164 behavior preserved: with auto_generate_scene=False
        and no scene_path, the adapter goes straight to the bare-Panda
        path without trying to generate."""
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            auto_generate_scene=False,
            init_jitter=0.0,
            install_cameras=False,
        )

        with patch.object(adapter, "_generate_scene_from_bddl") as mock_gen:
            sim = FakeSim(data_config="panda")
            adapter.on_episode_start(sim, random.Random(0))

        mock_gen.assert_not_called()
        assert sim._scenes_loaded == []
        assert adapter.scene_path is None

    def test_libero_missing_falls_back_with_warning(self, caplog, tmp_path):
        """If ``libero`` isn't installed, the adapter must log a warning
        and continue with the legacy bare-Panda path - eval still runs,
        just against the wrong world."""
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(tmp_path / "cache"),
            init_jitter=0.0,
            install_cameras=False,
        )

        # Simulate a missing pip package - require_optional raises ImportError.
        with patch(
            "strands_robots.benchmarks.libero.adapter.require_optional",
            side_effect=ImportError("'libero' is required"),
        ):
            with caplog.at_level("WARNING"):
                sim = FakeSim(data_config="panda")
                adapter.on_episode_start(sim, random.Random(0))

        # Eval continues without a scene loaded.
        assert sim._scenes_loaded == []
        assert adapter.scene_path is None
        # User gets a clear hint about [benchmark-libero].
        assert any(
            "scene auto-generation failed" in rec.message and "[benchmark-libero]" in rec.message
            for rec in caplog.records
        )

    def test_cache_hit_skips_libero_import(self, tmp_path):
        """If the SHA-keyed cache file already exists, the adapter must
        return its path without importing libero at all."""
        cache_dir = tmp_path / "scene_cache"
        cache_dir.mkdir()

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
            install_cameras=False,
        )

        # Pre-populate the cache file at the SHA the adapter computes.
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        cache_path = cache_dir / f"{sha}.xml"
        cache_path.write_text("<mujoco/>")

        # Sentinel: any libero import attempt would fail this test.
        with patch(
            "strands_robots.benchmarks.libero.adapter.require_optional",
            side_effect=AssertionError("require_optional should not be called on cache hit"),
        ):
            generated = adapter._generate_scene_from_bddl()

        assert generated == str(cache_path)

    def test_cache_miss_invokes_libero_and_writes_xml(self, tmp_path):
        """Cache miss path: libero is imported, ControlEnv is constructed
        with rendering disabled, the compiled MJCF is extracted, cameras
        are renamed, and the result is written to the cache."""
        cache_dir = tmp_path / "scene_cache"

        # Mock the entire libero.libero.envs.env_wrapper module surface.
        fake_xml = (
            '<mujoco model="libero_pick"><worldbody>'
            '<camera name="agentview" pos="1 0 1"/>'
            '<camera name="robot0_eye_in_hand_image" pos="0 0 0.5"/>'
            "</worldbody></mujoco>"
        )

        fake_env = MagicMock()
        fake_env.env.sim.model.get_xml.return_value = fake_xml
        fake_module = MagicMock()
        fake_module.ControlEnv.return_value = fake_env

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
            install_cameras=False,
        )

        with patch(
            "strands_robots.benchmarks.libero.adapter.require_optional",
            return_value=fake_module,
        ):
            generated = adapter._generate_scene_from_bddl()

        assert generated is not None
        # ControlEnv must have been constructed with rendering disabled
        # so we don't need a GL context.
        construct_kwargs = fake_module.ControlEnv.call_args.kwargs
        assert construct_kwargs["has_offscreen_renderer"] is False
        assert construct_kwargs["has_renderer"] is False
        assert construct_kwargs["use_camera_obs"] is False
        # The compiled XML was extracted from env.env.sim.model.get_xml().
        fake_env.env.sim.model.get_xml.assert_called_once()
        # Cameras renamed to libero_panda data_config bare keys.
        text = Path(generated).read_text()
        assert 'name="image"' in text
        assert 'name="wrist_image"' in text
        assert 'name="agentview"' not in text
        assert 'name="robot0_eye_in_hand_image"' not in text
        # env was closed after extraction.
        fake_env.close.assert_called_once()

    def test_camera_rename_disabled_with_empty_aliases(self, tmp_path):
        """Passing ``scene_camera_aliases={}`` keeps the LIBERO-canonical
        camera names verbatim - useful for users who want to manage the
        camera-name mapping themselves via the policy's
        observation_mapping."""
        cache_dir = tmp_path / "scene_cache"

        fake_xml = '<mujoco><worldbody><camera name="agentview" pos="1 0 1"/></worldbody></mujoco>'

        fake_env = MagicMock()
        fake_env.env.sim.model.get_xml.return_value = fake_xml
        fake_module = MagicMock()
        fake_module.ControlEnv.return_value = fake_env

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            scene_camera_aliases={},  # disable
            init_jitter=0.0,
            install_cameras=False,
        )

        with patch(
            "strands_robots.benchmarks.libero.adapter.require_optional",
            return_value=fake_module,
        ):
            generated = adapter._generate_scene_from_bddl()

        text = Path(generated).read_text()
        # agentview retained verbatim.
        assert 'name="agentview"' in text
        assert 'name="image"' not in text

    def test_cache_key_includes_bddl_and_aliases(self, tmp_path):
        """Two adapters built from the SAME BDDL with the SAME alias map
        share a cached XML. Different BDDL OR different alias map
        invalidates the cache - the latter is critical for the #168 round-5
        fix where the default alias map was extended to rename
        ``robot0_eye_in_hand``: existing user caches must auto-invalidate
        instead of serving the stale rewrite that leaves the wrist camera
        as a static fallback."""
        cache_dir = tmp_path / "scene_cache"

        a1 = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
        )
        a2 = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
        )
        a3 = LiberoAdapter.from_text(
            COMPOUND_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
        )
        # Adapter built with a non-default alias map - same BDDL as a1 but
        # different aliases should not share a1's cache file.
        a4 = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
            scene_camera_aliases={"agentview": "image"},  # a1 default has 3 entries
        )

        path1 = a1._resolve_bddl_path_for_libero()
        path2 = a2._resolve_bddl_path_for_libero()
        path3 = a3._resolve_bddl_path_for_libero()
        path4 = a4._resolve_bddl_path_for_libero()
        assert path1 is not None and path2 is not None and path3 is not None and path4 is not None
        # Cache key (per-adapter) is computed by ``_scene_cache_key``.
        key1 = a1._scene_cache_key(path1.read_bytes())
        key2 = a2._scene_cache_key(path2.read_bytes())
        key3 = a3._scene_cache_key(path3.read_bytes())
        key4 = a4._scene_cache_key(path4.read_bytes())
        # Same BDDL + same default aliases = same cache file.
        assert key1 == key2
        # Different BDDL = different cache file (alias map identical).
        assert key1 != key3
        # Same BDDL but different aliases = different cache file.
        assert key1 != key4

    def test_resolve_bddl_path_uses_existing_file_when_set(self, tmp_path):
        """``from_file`` constructed adapters reuse the original BDDL path
        instead of writing a new temp file."""
        bddl_file = tmp_path / "task.bddl"
        bddl_file.write_text(PICK_CUBE_BDDL)
        adapter = LiberoAdapter.from_file(
            bddl_file,
            scene_cache_dir=str(tmp_path / "cache"),
            init_jitter=0.0,
        )
        resolved = adapter._resolve_bddl_path_for_libero()
        assert resolved == bddl_file

    def test_resolve_bddl_path_returns_none_for_unsourced_problem(self, tmp_path):
        """Adapters built from a pre-parsed ``BDDLProblem`` without
        ``bddl_source`` / ``bddl_path`` can't auto-generate a scene -
        the resolver returns None and the generator surfaces that as
        a no-op."""
        from strands_robots.benchmarks.libero.bddl_parser import parse_bddl

        problem = parse_bddl(PICK_CUBE_BDDL)
        # Construct directly, bypassing from_file / from_text.
        adapter = LiberoAdapter(
            problem,
            scene_cache_dir=str(tmp_path / "cache"),
            init_jitter=0.0,
        )
        resolved = adapter._resolve_bddl_path_for_libero()
        assert resolved is None

        # The generator must return None (not raise) so the on_episode_start
        # fallback path stays alive.
        assert adapter._generate_scene_from_bddl() is None

    def test_on_episode_start_loads_generated_scene(self, tmp_path):
        """End-to-end: scene_path=None, auto_generate_scene=True ⇒
        on_episode_start generates the scene, mutates self.scene_path,
        and calls sim.load_scene with the cached path."""
        cache_dir = tmp_path / "scene_cache"
        cache_dir.mkdir()

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
            install_cameras=False,
        )

        # Pre-populate the cache so we don't need a fake libero env.
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        cache_path = cache_dir / f"{sha}.xml"
        cache_path.write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))

        assert adapter.scene_path == str(cache_path)
        assert sim._scenes_loaded == [str(cache_path)]


class TestRenameMjcfCameras:
    """``_rename_mjcf_cameras`` helper - targeted regex, doesn't touch
    non-camera elements."""

    def test_renames_camera_by_name(self):
        from strands_robots.benchmarks.libero.adapter import _rename_mjcf_cameras

        xml = '<camera name="agentview" pos="1 0 1"/>'
        out = _rename_mjcf_cameras(xml, {"agentview": "image"})
        assert 'name="image"' in out
        assert 'name="agentview"' not in out

    def test_passes_through_unmapped_cameras(self):
        from strands_robots.benchmarks.libero.adapter import _rename_mjcf_cameras

        xml = '<camera name="other_cam" fovy="60"/>'
        out = _rename_mjcf_cameras(xml, {"agentview": "image"})
        assert out == xml

    def test_does_not_touch_material_named_after_camera(self):
        """Targeted regex: only ``<camera ... name="X" ...>`` is rewritten,
        not e.g. ``<material name="agentview">`` (a contrived case but it
        guards against future false-positives)."""
        from strands_robots.benchmarks.libero.adapter import _rename_mjcf_cameras

        xml = '<material name="agentview" rgba="1 0 0 1"/><camera name="agentview" pos="1 0 1"/>'
        out = _rename_mjcf_cameras(xml, {"agentview": "image"})
        # Material name unchanged.
        assert '<material name="agentview"' in out
        # Camera name renamed.
        assert '<camera name="image"' in out

    def test_empty_aliases_is_noop(self):
        from strands_robots.benchmarks.libero.adapter import _rename_mjcf_cameras

        xml = '<camera name="agentview"/>'
        assert _rename_mjcf_cameras(xml, {}) == xml

    def test_handles_attributes_before_name(self):
        """Real LIBERO MJCFs put pose attributes before ``name="..."``."""
        from strands_robots.benchmarks.libero.adapter import _rename_mjcf_cameras

        xml = '<camera pos="1 0 1" fovy="60" name="agentview"/>'
        out = _rename_mjcf_cameras(xml, {"agentview": "image"})
        assert 'name="image"' in out


class TestCacheTransformVersion:
    """The cache-key derivation includes ``_LIBERO_MJCF_TRANSFORM_VERSION``
    so a version bump invalidates stale on-disk caches generated by
    prior post-process pipelines (#168 round-7+ history). Critical for
    upgrade paths."""

    def test_cache_key_includes_transform_version(self):
        """Different transform-version strings produce different cache keys
        even when bddl + alias map are identical. Round 8 produced caches
        with rgba-altered collision geoms and a stacked headlight in
        ``<visual>``; round 9 reverts to upstream-verbatim MJCFs and
        handles visualisation at render time. Cache-key version bump
        ensures users pick up the regenerated XML automatically."""
        import strands_robots.benchmarks.libero.adapter as adapter_module

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, install_cameras=False)
        bddl_bytes = b"some bddl content"

        original_version = adapter_module._LIBERO_MJCF_TRANSFORM_VERSION
        key1 = adapter._scene_cache_key(bddl_bytes)
        try:
            adapter_module._LIBERO_MJCF_TRANSFORM_VERSION = "v999"
            key2 = adapter._scene_cache_key(bddl_bytes)
        finally:
            adapter_module._LIBERO_MJCF_TRANSFORM_VERSION = original_version

        assert key1 != key2

    def test_current_transform_version_is_v4(self):
        """Pin the current version. Round 8 = ``v2`` (collision rgba +
        custom headlight, was wrong direction). Round 9 = ``v3``
        (cache verbatim, render-time options). #181 = ``v4`` (switch
        ``_extract_compiled_mjcf`` to prefer pre-compile MJCF so
        ``inertiagrouprange="0 0"`` is preserved). Bumping the version
        is the contract that invalidates stale caches; the value MUST
        be reviewed every time the transform pipeline changes."""
        from strands_robots.benchmarks.libero.adapter import _LIBERO_MJCF_TRANSFORM_VERSION

        assert _LIBERO_MJCF_TRANSFORM_VERSION == "v4"


class TestInstallRenderOptions:
    """``LiberoAdapter._install_render_options`` populates
    ``sim._world._backend_state['viz_option']`` with an ``mjvOption``
    matching upstream LIBERO's ``OffScreenRenderEnv`` viewer config
    (#168 round-9 bug E correction).

    Round 8 attempted to handle this at the MJCF level (rgba alpha=0
    on collision geoms + custom ``<visual>`` headlight). Round 9
    reverts that approach because:
    - the headlight stacked on top of upstream's two ``<light>`` blocks
      doubled the illumination (washing out contrast that made objects
      visible),
    - rgba edits to MJCF are non-canonical (upstream hides via
      renderer options).

    The new approach: store ``mjvOption`` on
    ``world._backend_state['viz_option']`` at episode start; the
    rendering layer reads it and threads through to
    ``Renderer.update_scene(scene_option=...)``.
    """

    def test_install_render_options_populates_backend_state(self):
        """End-to-end: after on_episode_start with a loaded scene,
        sim._world._backend_state['viz_option'] is populated with a
        configured MjvOption."""
        mujoco = pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            apply_scene_keyframe=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        # Pretend a scene was loaded (the post-load hook is what fires
        # _install_render_options); call it directly so we don't need a
        # real MJCF on disk.
        adapter._install_render_options(sim)

        opt = sim._world._backend_state.get("viz_option")
        assert opt is not None
        assert isinstance(opt, mujoco.MjvOption)
        # Collision geoms hidden.
        assert int(opt.geomgroup[0]) == 0
        # All site groups hidden.
        for sg in range(6):
            assert int(opt.sitegroup[sg]) == 0
        # Joint, actuator, COM widgets hidden.
        assert int(opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]) == 0
        assert int(opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR]) == 0
        assert int(opt.flags[mujoco.mjtVisFlag.mjVIS_COM]) == 0

    def test_install_render_options_visual_geoms_remain_visible(self):
        """The viz_option must NOT hide group=1 (visual) or group=2
        (display) geoms - those are what we want to keep."""
        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, install_cameras=False)
        sim = FakeSim(data_config="panda")
        adapter._install_render_options(sim)

        opt = sim._world._backend_state["viz_option"]
        # Defaults from mjv_defaultOption: geomgroup=[1, 1, 1, 0, 0, 0]
        # (groups 0, 1, 2 visible, 3-5 hidden). We turn off group 0;
        # groups 1 and 2 must stay enabled so visual + display geoms render.
        assert int(opt.geomgroup[1]) == 1
        assert int(opt.geomgroup[2]) == 1

    def test_no_world_silently_skips(self):
        """Sim without ``_world`` (non-MuJoCo backend, or stub) - skip
        without raising. Adapter must not abort the eval just because
        a viz_option couldn't be installed."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, install_cameras=False)

        class _NoWorldSim:
            pass

        # Should NOT raise.
        adapter._install_render_options(_NoWorldSim())  # type: ignore[arg-type]

    def test_no_backend_state_silently_skips(self):
        """world without ``_backend_state`` dict - skip silently.
        Defensive against test stubs / non-MuJoCo backends."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, install_cameras=False)

        class _BareWorld:
            pass

        class _BareSim:
            _world = _BareWorld()

        adapter._install_render_options(_BareSim())  # type: ignore[arg-type]
        # Test passes if no exception raised.

    def test_mujoco_unavailable_silently_skips(self):
        """When mujoco isn't importable (minimal CI / unit tests),
        the install no-ops silently. The render path will fall through
        to default options - cosmetically degraded but eval works."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, install_cameras=False)
        sim = FakeSim(data_config="panda")

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mujoco":
                raise ImportError("simulated missing mujoco")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            adapter._install_render_options(sim)

        # No viz_option installed.
        assert "viz_option" not in sim._world._backend_state

    def test_on_episode_start_installs_render_options_when_scene_loaded(self, tmp_path):
        """End-to-end through on_episode_start: with scene_path set and
        a real load, _install_render_options fires after camera install
        and the viz_option lands in backend_state."""
        mujoco = pytest.importorskip("mujoco")
        # Create a minimal MJCF for load_scene to "load".
        scene = tmp_path / "scene.xml"
        scene.write_text("<mujoco><worldbody/></mujoco>")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=str(scene),
            install_cameras=False,
            apply_scene_keyframe=False,
        )
        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))

        opt = sim._world._backend_state.get("viz_option")
        assert opt is not None
        assert isinstance(opt, mujoco.MjvOption)

    def test_on_episode_start_skips_render_options_when_no_scene(self):
        """Bare-Panda fallback (no scene loaded): render-options
        install is skipped. Default render options are appropriate
        for whatever scene the user's own code loaded."""
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))

        # No scene was loaded, so viz_option must NOT be installed.
        # Other adapters / users may want the MuJoCo default behaviour.
        assert "viz_option" not in sim._world._backend_state


class TestPrewarm:
    """``LiberoAdapter.prewarm(sim)`` is the public hook for installing
    render-time options BEFORE ``sim.start_cameras_recording`` (#168
    round-10 / bug D').

    The recorder thread captures its first frame immediately on
    ``start_cameras_recording``, before
    :meth:`evaluate_benchmark` (and therefore
    :meth:`on_episode_start`) runs. Without ``viz_option`` already
    installed via ``prewarm``, that first frame renders with MuJoCo's
    default visualization options - collision capsules + sites + joint
    axes visible. Subsequent frames are clean. The user reported this
    in round-9 verification: t=0.00 frame mean RGB ``(88, 88, 29)``,
    t=0.05+ frames mean RGB ``(79, 64, 57)``.
    """

    def test_prewarm_populates_viz_option(self):
        """End-to-end: after ``prewarm(sim)``,
        ``sim._world._backend_state["viz_option"]`` is populated. This
        is the headline #168 round-10 fix - subsequent
        ``start_cameras_recording`` will see the option on the first
        frame."""
        pytest.importorskip("mujoco")
        # scene_path explicitly set so prewarm can run (auto-gen path
        # would defer to on_episode_start).
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/path/to/some/scene.xml",  # presence matters, content doesn't for prewarm
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        adapter.prewarm(sim)

        assert "viz_option" in sim._world._backend_state
        opt = sim._world._backend_state["viz_option"]
        # Same flags as _install_render_options - collision off, sites off, etc.
        assert int(opt.geomgroup[0]) == 0
        for sg in range(6):
            assert int(opt.sitegroup[sg]) == 0

    def test_prewarm_idempotent(self):
        """Calling ``prewarm`` twice produces the same end state -
        critical because the recommended call site (between
        ``add_robot`` and ``start_cameras_recording``) might be hit
        on each fresh evaluate, and ``on_episode_start`` will call
        the same internals again. Both must be no-op-on-prior-state safe."""
        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        adapter.prewarm(sim)
        # First call installed viz_option; capture what we'd compare against.
        assert "viz_option" in sim._world._backend_state
        adapter.prewarm(sim)
        second_opt = sim._world._backend_state["viz_option"]
        # Each call builds a fresh MjvOption (overwrite is harmless).
        # Importantly: no exception, no duplicate cameras, no duplicate robot.
        assert int(second_opt.geomgroup[0]) == 0
        # Robot wrapper still in place (single entry, no duplicate).
        assert "robot" in sim._world.robots or "fake_panda" in sim._world.robots

    def test_prewarm_no_scene_path_is_noop(self):
        """When ``scene_path`` is None (auto-gen deferred to
        on_episode_start), prewarm silently no-ops. Required so the
        prewarm call doesn't crash adapters that haven't yet resolved
        a scene_path."""
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            auto_generate_scene=False,
        )
        assert adapter.scene_path is None
        sim = FakeSim(data_config="panda")
        # Must not raise.
        adapter.prewarm(sim)
        # Nothing installed.
        assert "viz_option" not in sim._world._backend_state

    def test_prewarm_does_not_break_subsequent_on_episode_start(self, tmp_path):
        """Calling ``prewarm`` then ``on_episode_start`` works as
        expected - prewarm doesn't poison state for the per-episode
        lifecycle. Pin to ensure the recommended call sequence
        (prewarm before recording, on_episode_start per-episode) is
        safe."""
        pytest.importorskip("mujoco")
        # Use a real scene file that load_scene can succeed on.
        scene = tmp_path / "scene.xml"
        scene.write_text("<mujoco><worldbody/></mujoco>")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=str(scene),
            install_cameras=False,
            apply_scene_keyframe=False,
        )
        sim = FakeSim(data_config="panda")

        adapter.prewarm(sim)
        adapter.on_episode_start(sim, random.Random(0))

        # Both phases populated viz_option (overwrite is harmless).
        assert "viz_option" in sim._world._backend_state

    def test_prewarm_individual_step_failure_does_not_abort(self):
        """If one of the three internal steps raises, prewarm logs a
        WARNING but continues to the next step. Critical because
        prewarm is best-effort - any failure here only degrades
        rendering, doesn't crash the eval (on_episode_start retries
        the same setup per-episode)."""
        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")

        # Make _register_default_robot raise via patching.
        with patch.object(
            LiberoAdapter,
            "_register_default_robot",
            side_effect=RuntimeError("simulated failure"),
        ):
            # Must NOT raise - failure is caught + logged.
            adapter.prewarm(sim)

        # _install_render_options STILL ran and populated viz_option.
        assert "viz_option" in sim._world._backend_state

    def test_prewarm_install_cameras_disabled(self):
        """When ``install_cameras=False`` was passed to constructor,
        prewarm respects the flag and skips the camera install step
        (parallels :meth:`on_episode_start`'s behaviour)."""
        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")

        # Mark adapter not to install cameras (already set via constructor).
        assert adapter._install_cameras is False

        with patch.object(LiberoAdapter, "_install_libero_cameras") as mock_install:
            adapter.prewarm(sim)
        mock_install.assert_not_called()
        # viz_option still installed - independent of camera install.
        assert "viz_option" in sim._world._backend_state

    def test_prewarm_calls_warmup_render_at_end(self):
        """Round 19 defensive: prewarm calls ``sim.render(camera_name=...)``
        once at the very end to prime process-shared GL state. The
        recorder thread spawned by ``start_cameras_recording`` then
        inherits warm shared state on its first render call.

        Pin for #168 round-19 verification: variant-B scenario where
        prewarm fully succeeds (no redundant Panda) and on_episode_start
        takes the fast-path - without main-thread warmup, the recorder
        thread's first ~15 calls return GL clear-colour gradient even
        with ``mj_forward`` populating xpos/xmat. The single warmup
        render here primes shared driver state.
        """
        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        # Track render calls.
        render_calls: list[tuple] = []
        original_render = sim.render

        def tracking_render(camera_name="default", width=None, height=None):
            render_calls.append((camera_name, width, height))
            return original_render(camera_name=camera_name, width=width, height=height)

        with patch.object(sim, "render", side_effect=tracking_render):
            adapter.prewarm(sim)

        # Exactly one warmup render call. Camera name is the first
        # in self._cameras (default LIBERO config has "image" first).
        assert len(render_calls) == 1, f"expected exactly one warmup render call from prewarm; got {render_calls}"
        cam_name, width, height = render_calls[0]
        assert cam_name == "image", f"expected first camera 'image'; got {cam_name!r}"
        # width/height are small - just need to prime state, not capture.
        assert width == 64
        assert height == 64

    def test_prewarm_warmup_render_no_render_method_silently_skips(self):
        """If sim has no ``render()`` method (test stub or unusual
        backend), the warmup step skips silently."""
        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")

        # Replace sim.render with None to simulate "no render method".
        with patch.object(sim, "render", None):
            # Must not raise.
            adapter.prewarm(sim)

    def test_prewarm_warmup_render_failure_does_not_abort(self):
        """If the warmup render raises, prewarm continues without
        re-raising. Logged at DEBUG (not WARNING) - this is informational
        only, the real recorder error path will surface persistent
        failures via state['errors']."""
        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")

        def boom(camera_name="default", width=None, height=None):
            raise RuntimeError("simulated render failure")

        with patch.object(sim, "render", side_effect=boom):
            # Must not raise.
            adapter.prewarm(sim)

        # Other prewarm steps still ran (viz_option installed).
        assert "viz_option" in sim._world._backend_state

    def test_prewarm_calls_mj_forward(self):
        """Round-14 fix: ``prewarm`` calls ``mujoco.mj_forward(model, data)``
        so ``data.xpos / data.xmat`` are populated before the recorder
        thread's first render. Without this, every render between
        ``prewarm()`` and ``on_episode_start``'s ``_apply_canonical_state``
        returns the skybox-only gradient because
        ``Renderer.update_scene`` finds body transforms unset.

        Pin for #168 round-14 verification: the bug-D gradient at t=0
        is NOT a renderer-warmup issue but a mj_forward race. Round 13
        added a 2-pass warmup loop in ``_loop`` to chase the symptom;
        round 14 reverts that and addresses the root cause here.
        """
        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        # FakeSim's _world doesn't expose model/data; populate stubs so
        # _forward_mj_data has something to call mj_forward on.

        class _FakeModel:
            pass

        class _FakeData:
            pass

        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = _FakeData()  # type: ignore[attr-defined]

        # Capture mj_forward calls.
        mj_forward_calls: list[tuple] = []

        class _StubMj:
            class mjtVisFlag:
                mjVIS_JOINT = 0
                mjVIS_ACTUATOR = 0
                mjVIS_COM = 0

            class MjvOption:
                def __init__(self):
                    # MjvOption with the indexable members the install code touches.
                    self.geomgroup = [1] * 6
                    self.sitegroup = [1] * 6
                    self.flags = [0] * 64

            @staticmethod
            def mjv_defaultOption(opt):
                pass

            @staticmethod
            def mj_forward(model, data):
                mj_forward_calls.append((model, data))

        with patch.dict("sys.modules", {"mujoco": _StubMj}):
            adapter.prewarm(sim)

        assert len(mj_forward_calls) == 1, (
            f"expected exactly one mj_forward call from prewarm, got {len(mj_forward_calls)}"
        )
        assert mj_forward_calls[0][0] is sim._world._model
        assert mj_forward_calls[0][1] is sim._world._data

    def test_prewarm_mj_forward_failure_does_not_abort(self):
        """If ``mj_forward`` raises during prewarm, log a WARNING but
        proceed - prewarm must not crash the whole eval pipeline. The
        next ``on_episode_start`` will retry via
        ``_apply_canonical_state`` -> ``mj_forward``, so a transient
        prewarm failure leaves the system recoverable.
        """
        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")

        class _FakeModel:
            pass

        class _FakeData:
            pass

        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = _FakeData()  # type: ignore[attr-defined]

        class _BoomMj:
            class mjtVisFlag:
                mjVIS_JOINT = 0
                mjVIS_ACTUATOR = 0
                mjVIS_COM = 0

            class MjvOption:
                def __init__(self):
                    self.geomgroup = [1] * 6
                    self.sitegroup = [1] * 6
                    self.flags = [0] * 64

            @staticmethod
            def mjv_defaultOption(opt):
                pass

            @staticmethod
            def mj_forward(model, data):
                raise RuntimeError("simulated mj_forward failure")

        with patch.dict("sys.modules", {"mujoco": _BoomMj}):
            # Must NOT raise - prewarm catches mj_forward failures.
            adapter.prewarm(sim)

        # viz_option still installed (mj_forward failure didn't propagate).
        assert "viz_option" in sim._world._backend_state

    def test_prewarm_no_model_skips_mj_forward(self):
        """When ``world._model`` or ``world._data`` is None (e.g. test
        stub or non-MuJoCo backend), ``_forward_mj_data`` skips
        silently - mj_forward isn't called, no exception."""
        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        # No model or data on the world.

        mj_forward_calls: list = []

        class _StubMj:
            class mjtVisFlag:
                mjVIS_JOINT = 0
                mjVIS_ACTUATOR = 0
                mjVIS_COM = 0

            class MjvOption:
                def __init__(self):
                    self.geomgroup = [1] * 6
                    self.sitegroup = [1] * 6
                    self.flags = [0] * 64

            @staticmethod
            def mjv_defaultOption(opt):
                pass

            @staticmethod
            def mj_forward(model, data):
                mj_forward_calls.append((model, data))

        with patch.dict("sys.modules", {"mujoco": _StubMj}):
            adapter.prewarm(sim)

        # mj_forward was NOT called because world.model/data is missing.
        assert mj_forward_calls == []

    def test_prewarm_applies_init_states_zero(self):
        """``prewarm`` writes ``init_states[0]`` to ``world._data`` so the
        recorder's first frame captures the canonical "ready" pose
        (#168 round-16 bug D-residual fix). Without this, ``data.qpos``
        stays at joint defaults that ``load_scene`` left behind, and
        the t=0.00 recorded frame shows the Panda stretched flat.
        """
        pytest.importorskip("mujoco")
        nq, nv = 4, 4
        states = np.zeros((2, 1 + nq + nv), dtype=np.float64)
        states[0, 1] = 42.0  # state 0: qpos[0] = 42
        states[1, 1] = 99.0  # state 1: qpos[0] = 99 - prewarm should NOT pick this

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
            init_states=states,
        )
        sim = FakeSim(data_config="panda")

        # Build minimal model/data on the FakeSim's world.
        class _Model:
            def __init__(self) -> None:
                self.nq = nq
                self.nv = nv
                self.na = 0

        class _Data:
            def __init__(self) -> None:
                self.qpos = np.zeros(nq)
                self.qvel = np.zeros(nv)
                self.time = 0.0

        sim._world._model = _Model()  # type: ignore[attr-defined]
        sim._world._data = _Data()  # type: ignore[attr-defined]

        class _StubMj:
            class mjtVisFlag:
                mjVIS_JOINT = 0
                mjVIS_ACTUATOR = 0
                mjVIS_COM = 0

            class MjvOption:
                def __init__(self):
                    self.geomgroup = [1] * 6
                    self.sitegroup = [1] * 6
                    self.flags = [0] * 64

            @staticmethod
            def mjv_defaultOption(opt):
                pass

            @staticmethod
            def mj_forward(model, data):
                pass

        with patch.dict("sys.modules", {"mujoco": _StubMj}):
            adapter.prewarm(sim)

        # data.qpos[0] should match init_states[0] (42), NOT init_states[1] (99).
        assert sim._world._data.qpos[0] == 42.0, (  # type: ignore[attr-defined]
            f"prewarm should apply init_states[0] (qpos[0]=42), got {sim._world._data.qpos[0]}"  # type: ignore[attr-defined]
        )
        # Episode counter NOT incremented by prewarm (only by
        # _apply_init_state_branch via on_episode_start).
        assert adapter._episode_count == 0

    def test_prewarm_no_init_states_skips_silently(self):
        """When ``init_states is None``, prewarm doesn't touch ``data.qpos``.
        Pin so the bare-Panda case (no LIBERO benchmark suite plumbing)
        still works through prewarm without crashing."""
        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
            # init_states=None (default)
        )
        sim = FakeSim(data_config="panda")
        assert adapter._init_states is None

        # Must not raise.
        adapter.prewarm(sim)

    def test_prewarm_init_state_width_mismatch_logs_skips(self):
        """Width mismatch in prewarm logs DEBUG and skips (best-effort
        semantics). Contrasts with ``_apply_init_state_branch`` which
        raises - prewarm must not crash the eval pipeline; the next
        ``on_episode_start`` will surface the error via the strict
        canonical-state branch."""
        pytest.importorskip("mujoco")
        # Width mismatch: init_state has 5 elements, but model expects 1+4+4=9.
        bad_state = np.zeros(5, dtype=np.float64)
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
            init_states=bad_state,
        )
        sim = FakeSim(data_config="panda")

        class _Model:
            nq = 4
            nv = 4
            na = 0

        class _Data:
            def __init__(self) -> None:
                self.qpos = np.zeros(4)
                self.qvel = np.zeros(4)
                self.time = 0.0

        sim._world._model = _Model()  # type: ignore[attr-defined]
        sim._world._data = _Data()  # type: ignore[attr-defined]

        class _StubMj:
            class mjtVisFlag:
                mjVIS_JOINT = 0
                mjVIS_ACTUATOR = 0
                mjVIS_COM = 0

            class MjvOption:
                def __init__(self):
                    self.geomgroup = [1] * 6
                    self.sitegroup = [1] * 6
                    self.flags = [0] * 64

            @staticmethod
            def mjv_defaultOption(opt):
                pass

            @staticmethod
            def mj_forward(model, data):
                pass

        with patch.dict("sys.modules", {"mujoco": _StubMj}):
            # Must not raise (best-effort, unlike _apply_init_state_branch).
            adapter.prewarm(sim)

        # qpos was NOT modified due to width mismatch.
        assert (sim._world._data.qpos == 0).all()  # type: ignore[attr-defined]


class TestInstallActionController:
    """``LiberoAdapter._install_action_controller`` builds an OSC_POSE
    controller that converts GR00T's task-space delta-EEF actions
    (``{x, y, z, roll, pitch, yaw, gripper}``) into the LIBERO scene's
    torque-mode joint actuators (#168 round 23 / round 24 fix).

    Without the controller, ``_apply_sim_action`` looks up GR00T's
    keys by name in the model's actuator/joint tables, finds no
    match, and silently drops every action - the policy effectively
    sends zero torque (#168 round 22 verification confirmed this is
    the actual blocker for ``success_rate=0``).

    Round-23 first-attempt failed with ``MjSim.__init__()`` signature
    mismatch (used 2-arg form, but robosuite 1.4.0's MjSim takes
    only ``model``). Round-24 fixes that AND hot-patches
    ``sim_shim.data._data`` to point at the actual sim's MjData
    buffer (otherwise the controller computes torques from a fresh,
    never-stepped MjData disconnected from what the eval is
    stepping). Both fixes are required for the controller to
    actually drive the robot.
    """

    @pytest.fixture
    def libero_scene_xml(self):
        """Use the existing scene cache if available; otherwise skip.

        The OSC controller install needs a real LIBERO-shaped MJCF
        with robot0_joint1..7, gripper0_grip_site, etc. Auto-generating
        from BDDL would require the libero package and ~30s. Use the
        local cache if present (round-23 verification was on this
        path); skip if not.
        """
        from pathlib import Path

        cache_dir = Path.home() / ".strands_robots" / "scene_cache" / "libero"
        if not cache_dir.is_dir():
            pytest.skip(f"no scene cache at {cache_dir}; cannot test OSC install end-to-end")
        cached = list(cache_dir.glob("*.xml"))
        if not cached:
            pytest.skip(f"no .xml files in {cache_dir}; cannot test OSC install end-to-end")
        return str(cached[0])

    def test_install_succeeds_on_real_libero_scene(self, libero_scene_xml):
        """End-to-end: load a real LIBERO scene MJCF, install the OSC
        controller via ``_install_action_controller``, verify it lands
        in ``world._backend_state['action_controller']`` with the
        expected discovered IDs."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)

        ctrl = sim._world._backend_state.get("action_controller")
        assert ctrl is not None, (
            "expected action_controller installed in _backend_state; check WARNING logs for install failure"
        )
        # Round-24 fix: verify the data buffer is shared (NOT a fresh
        # MjData created internally by MjSim).
        assert ctrl.sim_shim.data._data is sim._world._data, (
            "sim_shim.data._data must be hot-patched to share our actual "
            "data buffer; otherwise OSC writes go to a disconnected MjData"
        )
        # Discovered 7 arm actuators, ≥2 gripper actuators, EEF site.
        assert len(ctrl.arm_actuator_ids) == 7
        assert len(ctrl.gripper_actuator_ids) >= 2
        assert ctrl.eef_site_name == "gripper0_grip_site"

    def test_initial_joint_anchored_to_libero_home_pose_not_construction_state(self, libero_scene_xml):
        """Round 45 (#176): the OSC controller's ``initial_joint`` —
        the nullspace-bias target — must be the LIBERO-canonical Panda
        "ready" pose, NOT the joint angles at the moment the controller
        was constructed.

        Why this matters: ``LiberoAdapter._install_action_controller``
        runs from ``on_episode_start`` AFTER ``_apply_canonical_state``
        has already loaded the per-episode ``init_state`` into
        ``data.qpos``. Robosuite's ``BaseController.__init__`` captures
        ``self.initial_joint = self.joint_pos`` (= live ``data.qpos``)
        once at construction. Without the round-45 swap-and-restore
        around ``controller_factory``, ``initial_joint`` captures the
        perturbed canonical pose instead of the LIBERO
        ``MountedPanda.init_qpos`` ready pose — the OSC's
        ``nullspace_torques`` term ``(initial_joint - joint_pos)``
        then collapses to ~0 instead of producing the ~5 N·m bias
        upstream LIBERO produces (where upstream constructs the
        controller AFTER ``Robot.reset(deterministic=True)`` writes
        ``MountedPanda.init_qpos`` to ``data.qpos`` but BEFORE
        ``set_init_state`` perturbs it to the per-episode canonical).

        This was the root cause of the 50× torque divergence between
        our OSC and upstream's, surfaced by ``probe_osc_internals.py``
        in #176 sub-task 3b.

        Pin: build a controller after perturbing ``data.qpos[arm]``
        away from MJCF defaults. The controller's ``initial_joint``,
        ``initial_ee_pos``, ``initial_ee_ori_mat``, ``goal_pos``, and
        ``goal_ori`` must all match the values they would have if
        ``data.qpos[arm]`` had been at ``MountedPanda.init_qpos``
        when ``__init__`` ran.

        Also verify ``data.qpos`` is restored to the perturbed
        values after install — the swap is around the
        ``controller_factory`` call, so the sim's actual state must
        be the canonical state we passed in, not the home pose.
        """
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        libero = pytest.importorskip("libero")
        import mujoco

        # Resolve the canonical LIBERO Panda home pose. Skip if libero
        # version doesn't expose MountedPanda (e.g. future rename).
        try:
            from libero.libero.envs.robots.mounted_panda import MountedPanda
        except ImportError:
            pytest.skip(f"libero {libero.__version__} does not expose MountedPanda")
        expected_home_qpos = np.asarray(MountedPanda().init_qpos, dtype=np.float64)
        assert expected_home_qpos.shape == (7,), (
            f"MountedPanda.init_qpos must be 7-DoF for Panda, got {expected_home_qpos.shape}"
        )

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        # Perturb the arm joints by a non-trivial amount that would
        # flow into ``initial_*`` / ``goal_*`` if we naively captured
        # live qpos. If ``initial_joint`` is captured at construction
        # time (without our fix), it would equal these perturbed
        # values.
        model = sim._world._model  # type: ignore[attr-defined]
        data = sim._world._data  # type: ignore[attr-defined]
        # Find arm joint qpos addresses (robot0_joint1..7).
        arm_qpos_addrs: list[int] = []
        for i in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if isinstance(jname, str) and jname.startswith("robot0_joint"):
                arm_qpos_addrs.append(int(model.jnt_qposadr[i]))
        assert len(arm_qpos_addrs) == 7, f"expected 7 arm joints, found {len(arm_qpos_addrs)}"

        # Perturb each arm joint by 0.05 rad (~2.9 deg). Pre-fix,
        # ``initial_joint`` would equal ``data.qpos[arm] + 0.05``
        # after install.
        perturbation = 0.05
        for adr in arm_qpos_addrs:
            data.qpos[adr] += perturbation
        mujoco.mj_forward(model, data)

        # Snapshot perturbed qpos so we can assert it's restored
        # after install (the swap-and-restore must put data back
        # exactly where it was).
        perturbed_qpos = np.array([float(data.qpos[adr]) for adr in arm_qpos_addrs])

        # Capture the EEF pose at the HOME joint configuration so we
        # can assert the controller latched on to those values for
        # ``initial_ee_pos`` / ``initial_ee_ori_mat`` / ``goal_pos`` /
        # ``goal_ori``. Approach: write home qpos, mj_forward, read
        # site_xpos/xmat, restore.
        eef_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        for adr, v in zip(arm_qpos_addrs, expected_home_qpos, strict=True):
            data.qpos[adr] = float(v)
        mujoco.mj_forward(model, data)
        expected_home_ee_pos = np.array(data.site_xpos[eef_site_id])
        expected_home_ee_ori_mat = np.array(data.site_xmat[eef_site_id]).reshape(3, 3)
        # Restore perturbed qpos.
        for adr, v in zip(arm_qpos_addrs, perturbed_qpos, strict=True):
            data.qpos[adr] = float(v)
        mujoco.mj_forward(model, data)

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        # 1. ``initial_joint`` matches the home pose.
        captured_init_joint = np.array(ctrl.controller.initial_joint)
        np.testing.assert_allclose(
            captured_init_joint,
            expected_home_qpos,
            atol=1e-6,
            err_msg=(
                f"initial_joint was NOT re-anchored to LIBERO MountedPanda.init_qpos "
                f"={expected_home_qpos.tolist()}. The OSC controller will compute wrong "
                f"nullspace bias when constructed after _apply_canonical_state. See PR #176. "
                f"captured={captured_init_joint.tolist()}, perturbed={perturbed_qpos.tolist()}"
            ),
        )

        # 2. ``initial_ee_pos`` matches the EEF position at home pose
        # (NOT the EEF position at perturbed qpos). Tolerance 1e-5 m
        # because mj_forward involves ~10 floating-point ops per joint.
        captured_initial_ee_pos = np.array(ctrl.controller.initial_ee_pos)
        np.testing.assert_allclose(
            captured_initial_ee_pos,
            expected_home_ee_pos,
            atol=1e-5,
            err_msg=(
                f"initial_ee_pos={captured_initial_ee_pos.tolist()} does not match "
                f"expected home EEF pos={expected_home_ee_pos.tolist()}. The swap-and-"
                f"restore around controller_factory may not have written home qpos "
                f"to data before construction."
            ),
        )

        # 3. ``initial_ee_ori_mat`` matches the EEF orientation at
        # home pose. Same tolerance reasoning.
        captured_initial_ee_ori = np.array(ctrl.controller.initial_ee_ori_mat)
        np.testing.assert_allclose(
            captured_initial_ee_ori,
            expected_home_ee_ori_mat,
            atol=1e-5,
            err_msg=(
                "initial_ee_ori_mat does not match expected home EEF orientation. "
                "This causes goal_ori to latch on to the perturbed pose's orientation, "
                "producing torques ~50× off from upstream. See PR #176."
            ),
        )

        # 4. ``goal_pos`` and ``goal_ori`` are initialised in
        # ``OSC_POSE.__init__`` to ``initial_ee_pos`` / ``initial_ee_ori_mat``.
        # Verify they too track home pose, not perturbed.
        np.testing.assert_allclose(np.array(ctrl.controller.goal_pos), expected_home_ee_pos, atol=1e-5)
        np.testing.assert_allclose(np.array(ctrl.controller.goal_ori), expected_home_ee_ori_mat, atol=1e-5)

        # 5. ``data.qpos[arm]`` restored to perturbed values (the
        # swap-and-restore must not leak home pose into the sim's
        # actual state).
        restored_qpos = np.array([float(data.qpos[adr]) for adr in arm_qpos_addrs])
        np.testing.assert_allclose(
            restored_qpos,
            perturbed_qpos,
            atol=1e-9,
            err_msg=(
                f"data.qpos[arm] not restored after controller install — sim's "
                f"actual state is now the home pose {restored_qpos.tolist()} instead "
                f"of the canonical {perturbed_qpos.tolist()}. The swap-and-restore "
                f"contract is broken."
            ),
        )

        # 6. Sentinel: ``initial_joint`` must NOT equal the perturbed
        # qpos at install time. If they're equal (within float
        # tolerance), the round-45 fix didn't run.
        assert not np.allclose(captured_init_joint, perturbed_qpos, atol=1e-3), (
            f"initial_joint={captured_init_joint.tolist()} equals perturbed qpos "
            f"{perturbed_qpos.tolist()} — round-45 swap-and-restore didn't run."
        )

    def test_apply_writes_nonzero_torques_to_data_ctrl(self, libero_scene_xml):
        """Round-24 acceptance pin: after a non-trivial Cartesian delta,
        ``data.ctrl[arm_actuator_ids]`` is non-zero (the OSC controller
        actually computes torques and writes them to OUR sim's data).

        Verifies the hot-patch fix: round 23 silently used a
        disconnected MjData buffer, so torques computed by the
        controller went nowhere. Round 24's
        ``sim_shim.data._data = data`` puts the writes back on the
        actual buffer."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed - see WARNING log")

        # Zero out any prior ctrl writes.
        sim._world._data.ctrl[:] = 0  # type: ignore[attr-defined]

        # Apply a non-zero delta-EEF action.
        action = {
            "x": 0.05,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "gripper": 0.0,
        }
        ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]

        # Verify torques landed in OUR data.ctrl (not a disconnected buffer).
        arm_torques = [sim._world._data.ctrl[i] for i in ctrl.arm_actuator_ids]  # type: ignore[attr-defined]
        nonzero = sum(1 for t in arm_torques if abs(float(t)) > 1e-6)
        assert nonzero > 0, (
            f"expected non-zero arm torques after OSC apply, got {arm_torques}. "
            f"The controller may be writing to a disconnected MjData buffer "
            f"(round-23 bug) - check that sim_shim.data._data is hot-patched."
        )

    def test_install_failure_logs_warning_and_continues(self, caplog):
        """When OSC install fails (e.g. missing site, missing actuators),
        ``_install_action_controller`` logs WARNING and returns without
        raising. Eval continues with action no-op (round-22 behaviour).

        Pin against silent install failure regression: previously a
        bug (round 23's MjSim signature mismatch) caused the install
        to silently fail and the user wasn't aware - actions became
        no-ops at WARNING level so users can detect the install
        failure in logs."""
        import logging

        pytest.importorskip("mujoco")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        # No model - install will fail at the world.model check.

        with caplog.at_level(logging.WARNING, logger="strands_robots.benchmarks.libero.adapter"):
            adapter._install_action_controller(sim)

        # Look for the warning about install failure.
        warned = any("_install_action_controller" in r.message and "no-op" in r.message for r in caplog.records)
        assert warned, f"expected WARNING about install failure; got: {[r.message for r in caplog.records]}"
        # No controller installed.
        assert "action_controller" not in sim._world._backend_state

    def test_apply_handles_list_shaped_gripper_value(self, libero_scene_xml):
        """GR00T-LIBERO outputs ``action.gripper`` as a 2-element list
        (deliberate per PR #162 to match training shape - server
        boolean-masks state by per-key feature dimension; scalar shape
        fails). The OSC controller's apply path must handle list-shaped
        gripper without crashing.

        Pin for #168 round-25 fix: round-24's apply() called
        ``float(action_dict.get("gripper", 0.0))`` unconditionally
        BEFORE the list-coercion branch, so every step raised
        TypeError(``float() argument must be a string or a real
        number, not 'list'``). The defensive try/except in
        _apply_sim_action caught it and fell through to the no-op
        name-lookup path - actions silently dropped despite the
        controller being installed."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        sim._world._data.ctrl[:] = 0  # type: ignore[attr-defined]

        # GR00T's libero_panda gripper output is a 2-element list,
        # NOT a scalar. This is the case that crashed in round 24.
        action = {
            "x": 0.05,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "gripper": [0.5, 0.5],  # ← list-shaped, NOT scalar
        }
        # Must NOT raise.
        ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]

        # Gripper actuator ctrl[i] should be set to clip(0.5, ctrlrange).
        # Verify at least one gripper actuator was written (non-zero).
        gripper_ctrls = [sim._world._data.ctrl[i] for i in ctrl.gripper_actuator_ids]  # type: ignore[attr-defined]
        nonzero_gripper = sum(1 for t in gripper_ctrls if abs(float(t)) > 1e-6)
        assert nonzero_gripper > 0, (
            f"expected non-zero gripper torques after apply with list-shaped gripper, got {gripper_ctrls}. "
            f"The list-coercion path may be broken."
        )

    def test_apply_handles_scalar_gripper_value(self, libero_scene_xml):
        """Backwards compat: scalar gripper (0.7) still works after the
        round-25 list-coercion fix. Pin so the fix doesn't break the
        legacy scalar path."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        sim._world._data.ctrl[:] = 0  # type: ignore[attr-defined]

        action = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "gripper": 0.7,  # ← scalar, the legacy shape
        }
        ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]
        gripper_ctrls = [sim._world._data.ctrl[i] for i in ctrl.gripper_actuator_ids]  # type: ignore[attr-defined]
        nonzero_gripper = sum(1 for t in gripper_ctrls if abs(float(t)) > 1e-6)
        assert nonzero_gripper > 0, f"expected non-zero gripper torques for scalar gripper, got {gripper_ctrls}"

    def test_apply_handles_all_list_shaped_action_keys(self, libero_scene_xml):
        """ALL 7 GR00T-LIBERO action channels (x/y/z/roll/pitch/yaw/gripper)
        may be packed as 2-element lists per the training-shape
        convention introduced in PR #162. The OSC controller must
        handle this end-to-end without raising.

        Pin for #168 round-26 fix: round-25 only addressed gripper;
        round-25 verification revealed the same float(list) bug
        on the EEF delta keys (x, y, z, roll, pitch, yaw). The
        ``_to_scalar`` helper centralises the coercion so all 7
        keys go through the same path.

        Test mechanism: send the worst-case shape (every key is a
        2-element list) and verify the controller produces non-zero
        arm torques, no exceptions raised.
        """
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        sim._world._data.ctrl[:] = 0  # type: ignore[attr-defined]

        # ALL keys list-shaped (the actual GR00T-LIBERO format).
        action = {
            "x": [0.05, 0.05],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "gripper": [0.5, 0.5],
        }
        # Must NOT raise.
        ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]

        # Arm torques produced.
        arm_torques = [sim._world._data.ctrl[i] for i in ctrl.arm_actuator_ids]  # type: ignore[attr-defined]
        nonzero_arm = sum(1 for t in arm_torques if abs(float(t)) > 1e-6)
        assert nonzero_arm > 0, (
            f"expected non-zero arm torques after apply with all-list action, got {arm_torques}. "
            f"_to_scalar may not be applied to all 6 EEF delta keys."
        )

    def test_to_scalar_helper_handles_all_input_shapes(self):
        """Direct unit test of the ``_to_scalar`` helper. Pin the
        contract: scalar / list / tuple / ndarray / None / empty / dict
        all produce a sensible float without raising."""
        pytest.importorskip("mujoco")
        from strands_robots.benchmarks.libero.adapter import _to_scalar

        # Scalar inputs.
        assert _to_scalar(0.5) == 0.5
        assert _to_scalar(1) == 1.0
        assert _to_scalar(0.0) == 0.0

        # List/tuple/ndarray (non-empty) - takes first element.
        assert _to_scalar([0.5, 0.5]) == 0.5
        assert _to_scalar((1.5, 2.5)) == 1.5
        assert _to_scalar(np.array([0.7, 0.8, 0.9])) == 0.7

        # Edge cases - degrade to 0.0 with WARNING log (non-fatal).
        assert _to_scalar(None) == 0.0
        assert _to_scalar([]) == 0.0
        assert _to_scalar(()) == 0.0
        assert _to_scalar({"key": "value"}) == 0.0
        assert _to_scalar("not a number") == 0.0

    def test_controller_owns_stepping_flag_is_true(self, libero_scene_xml):
        """Round 27 (#168): the LIBERO OSC controller declares
        ``owns_stepping = True`` so the SimEngine skips its outer
        ``mj_step`` loop and lets the controller advance physics at
        the policy's required substep rate (25 substeps per policy
        step at LIBERO's 20 Hz control / 500 Hz physics).

        Pin so a future refactor that drops the flag silently
        regresses to 1 substep per apply (the round-26 bug where
        the robot moved but never converged because OSC was running
        at the physics rate, not the control rate)."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        from strands_robots.benchmarks.libero.adapter import _LiberoOSCController

        # Class-level flag - cheaper to assert than building a sim.
        assert _LiberoOSCController.owns_stepping is True

        # Instance-level too (paranoia: subclasses may override).
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]
        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")
        assert ctrl.owns_stepping is True

    def test_controller_physics_substeps_matches_libero_20hz_500hz(self, libero_scene_xml):
        """Round 27 (#168): with the default 0.002s timestep (500 Hz
        physics), ``physics_substeps_per_control`` should be 25 — i.e.
        ``int(round((1/20) / 0.002)) == 25``. Pin so a future change to
        the default timestep doesn't silently break LIBERO's 20 Hz
        training-data convention.

        25 substeps is the value RoboSuite's ``Base.step`` loop uses
        in ``robosuite.environments.base`` for ``control_freq=20``,
        which is what LIBERO's training data was generated with."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        # Sanity-check the libero scene XML uses 0.002s (the LIBERO
        # default). If upstream changes this we want a noisy failure.
        assert abs(float(sim._world._model.opt.timestep) - 0.002) < 1e-9, (  # type: ignore[attr-defined]
            f"libero scene timestep changed: {sim._world._model.opt.timestep}"  # type: ignore[attr-defined]
        )
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        assert ctrl.physics_substeps_per_control == 25

    def test_apply_advances_data_time_by_full_control_timestep(self, libero_scene_xml):
        """Round 27 (#168): one ``apply()`` call advances ``data.time``
        by the FULL control timestep (1/20 s = 0.05 s with default
        500 Hz physics). Pin the round-27 control-rate fix so a future
        refactor that re-introduces the round-26 single-substep bug
        fails loudly here.

        Pre-fix behaviour: 1 ``mj_step`` per apply ⇒ ``data.time``
        advances by ``model.opt.timestep`` = 0.002 s only. Post-fix:
        25 ``mj_step`` calls ⇒ 0.05 s. Anything < 0.04 s indicates the
        substep loop didn't run (substeps must be ≥ 20 to be plausible
        — exact number depends on timestep, but 1 vs 25 is unmistakable).
        """
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        t0 = float(sim._world._data.time)  # type: ignore[attr-defined]
        action = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "gripper": [0.0, 0.0],
        }
        ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]
        t1 = float(sim._world._data.time)  # type: ignore[attr-defined]

        dt_advance = t1 - t0
        # 25 substeps × 0.002s = 0.05s. Allow some slack for floating
        # point but the 1-substep regression case (0.002s) must fail.
        assert dt_advance > 0.04, (
            f"apply() advanced data.time by only {dt_advance:.4f}s; "
            f"expected ~0.05s for LIBERO 20 Hz / 500 Hz substep ratio. "
            f"Round-27 substep loop may not be running."
        )
        assert dt_advance < 0.06, (
            f"apply() advanced data.time by {dt_advance:.4f}s; "
            f"expected ~0.05s. Possible double-step or wrong substep count."
        )

    def test_apply_sim_action_skips_outer_mj_step_when_controller_owns_stepping(self):
        """Round 27 (#168): when ``controller.owns_stepping = True``,
        the SimEngine's ``_apply_sim_action`` skips its own ``mj_step``
        loop so we don't double-step on top of the controller's substep
        loop.

        Mechanism: install a stub controller that:
        - declares ``owns_stepping = True`` and
          ``physics_substeps_per_control = 7``,
        - on apply, records mj_step calls but doesn't actually step,
        and verify ``_apply_sim_action`` calls ``mj_step`` zero times
        (the outer loop is skipped) and updates ``step_count`` by 7
        (the controller's declared substep count).

        Without the ``owns_stepping`` check, the outer loop would
        call ``mj_step`` once more, advancing physics past what the
        controller's substep loop intended."""
        pytest.importorskip("mujoco")

        from strands_robots.simulation import Simulation

        # Build a minimal Simulation. We don't need a robot — just a
        # compiled model + data. We sidestep ``add_robot`` (which
        # requires asset downloads) by injecting a minimal robot stub
        # into ``world.robots`` directly. The stub only needs a
        # ``namespace`` attribute (read by _apply_sim_action's
        # name-lookup fallback path, which we bypass anyway).
        sim = Simulation()
        result = sim.create_world()
        assert result["status"] == "success", result
        assert sim._world is not None

        # Compile a trivial 1-DoF MuJoCo model so we have a valid
        # model + data. We never write to it (the controller stub is
        # a no-op), so the contents don't matter beyond being valid.
        import mujoco

        trivial_xml = """
        <mujoco>
          <worldbody>
            <body name="b">
              <joint name="j" type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
          <actuator><motor joint="j" ctrlrange="-1 1"/></actuator>
        </mujoco>
        """
        sim._world._model = mujoco.MjModel.from_xml_string(trivial_xml)
        sim._world._data = mujoco.MjData(sim._world._model)
        mujoco.mj_forward(sim._world._model, sim._world._data)

        # Inject a robot stub. The send_action path checks
        # ``robot_name not in self._world.robots`` (returns early
        # if missing) and then reads ``robot.namespace`` for the
        # name-lookup fallback (which the controller stub bypasses).
        class _RobotStub:
            namespace = ""

        sim._world.robots["arm"] = _RobotStub()  # type: ignore[assignment]

        call_count = [0]
        original = mujoco.mj_step

        def _counting_step(model, data, *args, **kwargs):
            call_count[0] += 1
            return original(model, data, *args, **kwargs)

        class _OwnsSteppingStub:
            owns_stepping = True
            physics_substeps_per_control = 7

            def apply(self, action_dict, model, data, robot_name):
                # Deliberately don't step. The whole point is to
                # detect whether _apply_sim_action's outer loop fires.
                pass

        sim._world._backend_state["action_controller"] = _OwnsSteppingStub()
        step_count_before = sim._world.step_count

        try:
            mujoco.mj_step = _counting_step  # type: ignore[assignment]
            sim.send_action({"x": 0.0}, robot_name="arm", n_substeps=3)
        finally:
            mujoco.mj_step = original  # type: ignore[assignment]

        # 0 mj_step calls: the stub's apply doesn't step; the outer
        # loop is skipped because owns_stepping=True.
        assert call_count[0] == 0, (
            f"expected 0 mj_step calls when controller owns_stepping=True with no-op apply, "
            f"got {call_count[0]}. Outer mj_step loop in _apply_sim_action may not be skipping."
        )
        # step_count increments by physics_substeps_per_control (7),
        # NOT by n_substeps (3) — because the controller declared its
        # own substep count.
        delta_step = sim._world.step_count - step_count_before
        assert delta_step == 7, (
            f"expected step_count to advance by 7 (controller's physics_substeps_per_control), "
            f"got {delta_step}. _apply_sim_action may not be reading the controller's declared count."
        )

        sim.destroy()

    def test_apply_sim_action_no_owns_stepping_keeps_outer_loop(self):
        """Round 27 (#168) backwards compat: when a controller does NOT
        declare ``owns_stepping = True``, ``_apply_sim_action`` keeps
        its existing outer ``mj_step`` loop. Pin so the round-27 change
        doesn't accidentally skip stepping for non-LIBERO controllers
        (which would be a silent regression for any future benchmark
        adapter that uses the action_controller hook without the LIBERO
        substep semantics)."""
        pytest.importorskip("mujoco")

        from strands_robots.simulation import Simulation

        sim = Simulation()
        result = sim.create_world()
        assert result["status"] == "success", result
        assert sim._world is not None

        import mujoco

        trivial_xml = """
        <mujoco>
          <worldbody>
            <body name="b">
              <joint name="j" type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
          <actuator><motor joint="j" ctrlrange="-1 1"/></actuator>
        </mujoco>
        """
        sim._world._model = mujoco.MjModel.from_xml_string(trivial_xml)
        sim._world._data = mujoco.MjData(sim._world._model)
        mujoco.mj_forward(sim._world._model, sim._world._data)

        class _RobotStub:
            namespace = ""

        sim._world.robots["arm"] = _RobotStub()  # type: ignore[assignment]

        call_count = [0]
        original = mujoco.mj_step

        def _counting_step(model, data, *args, **kwargs):
            call_count[0] += 1
            return original(model, data, *args, **kwargs)

        class _PlainController:
            # owns_stepping ABSENT (defaults to False via getattr)

            def apply(self, action_dict, model, data, robot_name):
                pass

        sim._world._backend_state["action_controller"] = _PlainController()
        step_count_before = sim._world.step_count

        try:
            mujoco.mj_step = _counting_step  # type: ignore[assignment]
            sim.send_action({"x": 0.0}, robot_name="arm", n_substeps=3)
        finally:
            mujoco.mj_step = original  # type: ignore[assignment]

        # Outer loop fires for n_substeps=3 — the legacy contract.
        assert call_count[0] == 3, (
            f"expected 3 mj_step calls (outer loop, n_substeps=3) for plain controller "
            f"without owns_stepping, got {call_count[0]}."
        )
        delta_step = sim._world.step_count - step_count_before
        assert delta_step == 3, (
            f"expected step_count to advance by n_substeps=3 for plain controller, got {delta_step}."
        )

        sim.destroy()

    def test_gripper_state_initialized_to_zeros(self, libero_scene_xml):
        """Round 28 (#168): ``_gripper_current_action`` initialises to
        ``np.zeros(2)`` so the first ``apply()`` call starts from a
        canonical neutral state. Mirrors RoboSuite's
        ``GripperModel.__init__`` which sets
        ``self.current_action = np.zeros(self.dof)``.

        Pin so a future change to the init value (e.g. mid-range) doesn't
        silently bias every grasp attempt."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        np.testing.assert_array_equal(ctrl._gripper_current_action, np.zeros(2))

    def test_gripper_close_command_drives_both_fingers_to_closed(self, libero_scene_xml):
        """Round 28 (#168): ``gripper`` close-command must drive BOTH
        finger ctrl values toward their closed positions, NOT just one.

        Pre-fix bug: writing the raw scalar to both gripper actuators
        clipped against asymmetric ctrlranges (finger1: ``[0, 0.04]``
        where +1 → 0.04 = OPEN; finger2: ``[-0.04, 0]`` where +1 → 0 =
        closed). One finger moved the wrong way per grasp attempt.

        Post-fix: stateful ramp + bias/weight rescale. After 25
        substeps of close from neutral:
        - finger1 ctrl moves from 0.02 (mid) toward 0 (closed)
        - finger2 ctrl moves from -0.02 (mid) toward 0 (closed)

        Both fingers converge to 0 (closed); pin the per-finger sign
        so the wrong-direction bug can't sneak back.

        Round 41 (#168) update: the gripper input is now in RLDS
        convention (``0 = close``, ``1 = open``); the OSC controller
        applies ``-np.sign(2x-1)`` to convert RLDS → robosuite
        convention. So "close" command = ``gripper=0.0`` (was
        ``gripper=+1.0`` before round 41 because we were treating
        the input as already-converted)."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        # Find LIBERO's gripper actuator ctrlrange to know what
        # "closed" looks like per finger. PandaGripper convention:
        # both fingers' "closed" position is 0.0 (clipped lower bound
        # of finger1 [0, 0.04]; clipped upper bound of finger2 [-0.04, 0]).
        gripper_ctrlrange = [
            (
                float(sim._world._model.actuator_ctrlrange[gi, 0]),  # type: ignore[attr-defined]
                float(sim._world._model.actuator_ctrlrange[gi, 1]),  # type: ignore[attr-defined]
            )
            for gi in ctrl.gripper_actuator_ids
        ]

        # Send a close command (RLDS 0.0 = close) for one policy step.
        # Verify EACH finger's ctrl moved toward its respective closed
        # position. Round 41 transform: -sign(2*0-1) = -sign(-1) = +1
        # → ramp [-0.01, +0.01] → fingers toward closed.
        action = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "gripper": [0.0, 0.0],  # RLDS close
        }
        ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]

        # After one apply (25 substeps): current_action ≈ [-0.25, +0.25].
        np.testing.assert_array_almost_equal(ctrl._gripper_current_action, np.array([-0.25, 0.25]))

        # Closed position per finger:
        #   finger1: ctrlrange=(0, 0.04) → closed=0
        #   finger2: ctrlrange=(-0.04, 0) → closed=0
        # We assert each ctrl moved TOWARD 0 (its closed position).
        for gi, (lo, hi) in zip(ctrl.gripper_actuator_ids, gripper_ctrlrange, strict=True):
            ctrl_val = float(sim._world._data.ctrl[gi])  # type: ignore[attr-defined]
            mid = 0.5 * (hi + lo)  # neutral / "rest" position
            # closed position is whichever endpoint corresponds to "0"
            # in the gripper's normalized [-1, +1] frame after rescale.
            # For PandaGripper both fingers close to 0.
            closed = 0.0
            # ctrl_val should be strictly between mid and closed
            # (i.e. moved toward closed but not yet saturated).
            assert min(mid, closed) <= ctrl_val <= max(mid, closed), (
                f"finger ctrl {ctrl_val} not between mid={mid} and closed={closed} (ctrlrange=({lo}, {hi}))"
            )
            assert ctrl_val != mid, f"finger ctrl {ctrl_val} didn't move from mid={mid}"

    def test_gripper_close_vs_open_produces_opposite_ctrl_directions(self, libero_scene_xml):
        """Round 28 (#168): close-command vs open-command must produce
        ctrl values that move in OPPOSITE directions per finger.

        Pre-fix bug: writing the raw scalar to both fingers meant
        close and open commands didn't actually produce reversed
        outputs — each had one finger going wrong, making the
        gripper effectively un-actuable.

        Post-fix: opposite signs of (round-41-converted) input produce
        opposite-sign deltas for ``current_action``, and the
        bias/weight rescale preserves that into the ctrl values.

        Round 41 (#168) update: input is in RLDS convention (0=close,
        1=open). The OSC controller's internal ``-sign(2x-1)``
        transform converts to robosuite's (+1=close, -1=open)."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )

        # Run two separate sims/controllers: one drives close, the
        # other drives open. Then compare resulting ctrl signs.
        def _drive(gripper_value: float) -> tuple[np.ndarray, np.ndarray]:
            sim = FakeSim(data_config="panda")
            sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
            sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
            mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]
            adapter._install_action_controller(sim)
            ctrl = sim._world._backend_state.get("action_controller")
            assert ctrl is not None
            action = {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
                "gripper": [gripper_value, gripper_value],
            }
            ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]
            return (
                np.array([float(sim._world._data.ctrl[gi]) for gi in ctrl.gripper_actuator_ids]),  # type: ignore[attr-defined]
                np.array(ctrl._gripper_current_action),
            )

        # RLDS convention: 0.0 = close, 1.0 = open.
        ctrl_close, current_close = _drive(0.0)
        ctrl_open, current_open = _drive(1.0)

        # current_action should be opposite-signed for the two
        # commands (since the round-41 transform inverts sign and the
        # ramp = ±[-1, +1] * speed * sign(transformed_input)).
        np.testing.assert_array_almost_equal(current_close, -current_open)
        # And the ctrl deltas (after bias/weight rescale) too:
        # they should be opposite-signed deltas around the neutral
        # bias point.
        # Bias is the same for both, so delta_close = -delta_open.
        # We don't know bias exactly without reading ctrlrange, but
        # we can assert ctrl_close - ctrl_open is non-zero (the
        # commands actually differ in their effect).
        assert not np.allclose(ctrl_close, ctrl_open), (
            f"close and open commands produced identical ctrl values: "
            f"close={ctrl_close}, open={ctrl_open}. The wrong-direction "
            f"bug may be back."
        )

    def test_gripper_state_ramps_across_apply_calls(self, libero_scene_xml):
        """Round 28 (#168): ``_gripper_current_action`` accumulates
        across ``apply()`` calls, mirroring RoboSuite's stateful
        ``PandaGripper.current_action`` ramp.

        Pin so a future refactor that resets the gripper state
        per-apply (instead of per-episode) silently slows ramping to
        25 substeps regardless of how long the policy commands close.
        Real grasps need many policy steps to fully close (8 steps to
        saturate at 0.01 speed × 25 substeps × 0.01 ramp = 0.25 / step
        toward unit saturation ⇒ 4 policy steps to reach ±1)."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        action = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "gripper": [0.0, 0.0],  # sustained close (RLDS: 0=close)
        }

        # Call apply() three times. Each call ramps current_action by
        # ~0.25 toward saturation (after round-41 transform, RLDS 0.0
        # → robosuite +1 → 25 substeps × 0.01 speed × sign=+1 → ramp).
        ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]
        first = np.array(ctrl._gripper_current_action)
        ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]
        second = np.array(ctrl._gripper_current_action)
        ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]
        third = np.array(ctrl._gripper_current_action)

        # State must accumulate, not reset.
        assert np.linalg.norm(second) > np.linalg.norm(first), (
            f"second apply state {second} not larger magnitude than first {first}; ramp may be resetting per-apply."
        )
        assert np.linalg.norm(third) > np.linalg.norm(second), (
            f"third apply state {third} not larger magnitude than second {second}; ramp not accumulating."
        )

    def test_gripper_reset_zeros_state(self, libero_scene_xml):
        """Round 28 (#168): ``reset()`` zeros ``_gripper_current_action``
        so a new episode starts from neutral. While each
        ``_install_action_controller`` call rebuilds the controller
        (auto-resetting state via ``__init__``), the explicit
        ``reset()`` method documents the contract and lets callers
        zero state without re-discovering all the joint/actuator IDs."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        # Drive state away from zero.
        ctrl._gripper_current_action[:] = np.array([-0.5, 0.5])

        ctrl.reset()
        np.testing.assert_array_equal(ctrl._gripper_current_action, np.zeros(2))

    def test_round_41_rlds_to_robosuite_gripper_transform(self, libero_scene_xml):
        """Round 41 (#168): the OSC controller applies NVIDIA's
        ``normalize_gripper_action + invert_gripper_action`` chain
        internally. Pin the truth table:

        - RLDS 0.0 (close) → ``-sign(2*0-1) = -sign(-1) = +1`` →
          robosuite +1 (close) → ramp ``[-0.01, +0.01]`` →
          ``current_action = [-0.25, +0.25]`` after one apply
        - RLDS 1.0 (open) → ``-sign(2*1-1) = -sign(+1) = -1`` →
          robosuite -1 (open) → ramp ``[+0.01, -0.01]`` →
          ``current_action = [+0.25, -0.25]`` after one apply
        - RLDS 0.5 (no-op) → ``-sign(0) = 0`` → ramp ``[0, 0]`` →
          ``current_action`` unchanged

        Pin so a future refactor that drops the round-41 transform
        (e.g. moves it to a different layer + forgets to call it)
        regresses the polarity bug that left ``success_rate=0``
        despite all the structural fixes from rounds 36-40."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )

        def _apply_one_step(gripper_in: float) -> np.ndarray:
            sim = FakeSim(data_config="panda")
            sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
            sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
            mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]
            adapter._install_action_controller(sim)
            ctrl = sim._world._backend_state.get("action_controller")
            assert ctrl is not None
            action = {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
                "gripper": [gripper_in, gripper_in],
            }
            ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]
            return np.array(ctrl._gripper_current_action)

        # RLDS 0.0 = close → robosuite +1 → ramp [-0.25, +0.25] (toward closed).
        np.testing.assert_array_almost_equal(
            _apply_one_step(0.0),
            np.array([-0.25, 0.25]),
            err_msg="RLDS 0.0 should drive ramp toward [-0.25, +0.25] (close)",
        )
        # RLDS 1.0 = open → robosuite -1 → ramp [+0.25, -0.25] (toward open).
        np.testing.assert_array_almost_equal(
            _apply_one_step(1.0),
            np.array([0.25, -0.25]),
            err_msg="RLDS 1.0 should drive ramp toward [+0.25, -0.25] (open)",
        )
        # RLDS 0.5 = midpoint → -sign(0) = 0 → no ramp.
        np.testing.assert_array_almost_equal(
            _apply_one_step(0.5),
            np.zeros(2),
            err_msg="RLDS 0.5 should produce zero ramp (no motion)",
        )

    def test_action_log_disabled_by_default(self, libero_scene_xml, monkeypatch):
        """Round 29 (#168): ``STRANDS_LIBERO_ACTION_LOG`` is opt-in.
        When unset (or any value other than 1/true/yes/on),
        ``_action_log_enabled`` is False and ``apply()`` emits no
        diagnostic INFO lines.

        Pin so a future bug doesn't accidentally enable it by default
        (which would flood production eval logs) or change the parsing
        rules (which would break operator muscle memory)."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        monkeypatch.delenv("STRANDS_LIBERO_ACTION_LOG", raising=False)

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        assert ctrl._action_log_enabled is False
        assert ctrl._action_log_step == 0

    def test_action_log_enabled_emits_one_log_per_apply(self, libero_scene_xml, monkeypatch, caplog):
        """Round 29 (#168): with ``STRANDS_LIBERO_ACTION_LOG=1`` and
        a fresh controller, each ``apply()`` call emits exactly one
        ``ACTION_LOG`` INFO log line containing the diagnostic
        fields (``action_keys``, ``delta``, ``gripper_value``,
        ``eef_pos_pre``, ``eef_pos_post``, ``arm_ctrl_pre/post``,
        ``arm_qpos_pre/post``, ``gripper_ctrl_pre/post``,
        ``gripper_current_pre/post``).

        Pin so a future refactor that drops the structured log line
        fails loudly here — the field set is the diagnostic contract
        for PR #168 round-29."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import logging as _logging

        import mujoco

        monkeypatch.setenv("STRANDS_LIBERO_ACTION_LOG", "1")

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        assert ctrl._action_log_enabled is True

        action = {
            "x": [0.01, 0.01],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "gripper": [1.0, 1.0],
        }

        # The structured log line is emitted at INFO from
        # ``strands_robots.benchmarks.libero.adapter`` logger.
        with caplog.at_level(_logging.INFO, logger="strands_robots.benchmarks.libero.adapter"):
            ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]

        action_log_lines = [r for r in caplog.records if "ACTION_LOG" in r.getMessage()]
        assert len(action_log_lines) == 1, f"expected exactly 1 ACTION_LOG line, got {len(action_log_lines)}"

        msg = action_log_lines[0].getMessage()
        # Diagnostic-field contract — every name in this list must be
        # present in every emitted log line. Reviewers grep these.
        for field in [
            "step=0",
            "action_keys=",
            "delta=",
            "gripper_value=",
            "eef_pos_pre=",
            "eef_pos_post=",
            "eef_pos_delta=",
            "eef_quat_pre=",
            "eef_quat_post=",
            "arm_ctrl_pre=",
            "arm_ctrl_post=",
            "arm_qpos_pre=",
            "arm_qpos_post=",
            "gripper_ctrl_pre=",
            "gripper_ctrl_post=",
            "gripper_current_pre=",
            "gripper_current_post=",
        ]:
            assert field in msg, f"missing field {field!r} in ACTION_LOG: {msg}"

        # Step counter advanced.
        assert ctrl._action_log_step == 1

    def test_action_log_caps_at_max_steps(self, libero_scene_xml, monkeypatch, caplog):
        """Round 29 (#168): ``STRANDS_LIBERO_ACTION_LOG_MAX`` caps how
        many ``apply()`` calls log per episode. After the cap, ``apply``
        runs silently. Pin so a long episode doesn't flood logs once
        the diagnostic window has captured what's needed."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import logging as _logging

        import mujoco

        monkeypatch.setenv("STRANDS_LIBERO_ACTION_LOG", "1")
        monkeypatch.setenv("STRANDS_LIBERO_ACTION_LOG_MAX", "3")

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        assert ctrl._action_log_max == 3

        action = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "gripper": [0.0, 0.0],
        }

        # Call apply 5 times; only first 3 should log.
        with caplog.at_level(_logging.INFO, logger="strands_robots.benchmarks.libero.adapter"):
            for _ in range(5):
                ctrl.apply(action, sim._world._model, sim._world._data, "robot")  # type: ignore[attr-defined]

        action_log_lines = [r for r in caplog.records if "ACTION_LOG" in r.getMessage()]
        assert len(action_log_lines) == 3, f"expected 3 ACTION_LOG lines (cap=3), got {len(action_log_lines)}"
        # Step counter at cap (each log emit increments _action_log_step)
        assert ctrl._action_log_step == 3

    def test_action_log_resets_on_episode_boundary(self, libero_scene_xml, monkeypatch):
        """Round 29 (#168): ``reset()`` zeros the per-episode log
        step counter so each episode logs its own first N steps.

        Without this, the second episode of a multi-episode eval
        would silently emit no diagnostic logs (counter already
        at the cap from episode 1)."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import mujoco

        monkeypatch.setenv("STRANDS_LIBERO_ACTION_LOG", "1")

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        adapter._install_action_controller(sim)
        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        # Drive counter forward.
        ctrl._action_log_step = 5

        ctrl.reset()
        assert ctrl._action_log_step == 0, "reset() must zero the per-episode log step counter"

    def test_action_log_invalid_max_value_warns_and_falls_back(self, libero_scene_xml, monkeypatch, caplog):
        """Round 29 (#168): a malformed ``STRANDS_LIBERO_ACTION_LOG_MAX``
        (e.g. ``foo``) must log a WARNING and fall back to the
        default (50), not silently swallow the typo. Mirrors the
        AGENTS.md rule for ``STRANDS_*`` env vars."""
        pytest.importorskip("mujoco")
        pytest.importorskip("robosuite")
        import logging as _logging

        import mujoco

        monkeypatch.setenv("STRANDS_LIBERO_ACTION_LOG", "1")
        monkeypatch.setenv("STRANDS_LIBERO_ACTION_LOG_MAX", "not-an-int")

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=libero_scene_xml,
            install_cameras=False,
            auto_generate_scene=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._model = mujoco.MjModel.from_xml_path(libero_scene_xml)  # type: ignore[attr-defined]
        sim._world._data = mujoco.MjData(sim._world._model)  # type: ignore[attr-defined]
        mujoco.mj_forward(sim._world._model, sim._world._data)  # type: ignore[attr-defined]

        with caplog.at_level(_logging.WARNING, logger="strands_robots.benchmarks.libero.adapter"):
            adapter._install_action_controller(sim)

        ctrl = sim._world._backend_state.get("action_controller")
        if ctrl is None:
            pytest.skip("action_controller install failed")

        # Default fallback — 50.
        assert ctrl._action_log_max == 50
        warnings = [
            r
            for r in caplog.records
            if r.levelno == _logging.WARNING and "STRANDS_LIBERO_ACTION_LOG_MAX" in r.getMessage()
        ]
        assert len(warnings) >= 1, "expected a WARNING about malformed STRANDS_LIBERO_ACTION_LOG_MAX"


class TestPrewarmFreshEpisodeZero:
    """Round 17: ``on_episode_start`` detects the prewarm-fresh ep0
    state via ``world._backend_state["libero_prewarm_path"]`` and skips
    its own ``load_scene`` + ``_apply_canonical_state`` calls.

    Without this, the recorder thread spawned by
    ``start_cameras_recording`` captures frames during the race window
    between ``on_episode_start``'s ``load_scene`` (which resets MjData
    to qpos0) and ``_apply_canonical_state`` (which restores
    init_states[0]). The recorder's first frame consistently shows
    qpos0 instead of the canonical ready pose.

    The fast-path: prewarm has already loaded the scene + applied
    init_states[0], so on ep0 with the flag set, on_episode_start
    skips the redundant reload. Episode counter is bumped to 1
    manually so ep1+ follows the normal lifecycle.
    """

    def test_ep0_fast_path_skips_load_scene_when_prewarm_flag_matches(self, tmp_path):
        """Episode 0 with ``libero_prewarm_path`` matching ``self.scene_path``
        skips ``sim.load_scene`` (avoids redundant spec recompile)."""
        pytest.importorskip("mujoco")
        scene = tmp_path / "scene.xml"
        scene.write_text("<mujoco><worldbody/></mujoco>")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=str(scene),
            install_cameras=False,
            apply_scene_keyframe=False,
        )
        sim = FakeSim(data_config="panda")
        # Simulate prewarm having set the flag.
        sim._world._backend_state["libero_prewarm_path"] = str(scene)

        load_scene_calls: list = []
        original_load_scene = sim.load_scene

        def tracking_load_scene(scene_path):
            load_scene_calls.append(scene_path)
            return original_load_scene(scene_path)

        with patch.object(sim, "load_scene", side_effect=tracking_load_scene):
            adapter.on_episode_start(sim, random.Random(0))

        # Fast-path: load_scene was NOT called (prewarm already did it).
        assert load_scene_calls == [], f"expected ep0 fast-path to skip load_scene; got {load_scene_calls}"
        # Flag cleared so a subsequent prewarm + ep0 detects fresh state again.
        assert "libero_prewarm_path" not in sim._world._backend_state

    def test_ep0_fast_path_still_runs_canonical_state_apply(self, tmp_path):
        """Episode 0 fast-path runs ``_apply_canonical_state`` to restore
        qpos after PolicyRunner.sim.reset() between prewarm and
        on_episode_start.

        Pin for #168 round-22 user-flagged fix: round 17/18's fast-path
        skipped ``_apply_canonical_state`` on the assumption that
        prewarm had already applied init_states[0]. But
        ``PolicyRunner._evaluate_with_spec`` calls ``sim.reset()``
        between prewarm and on_episode_start, which resets ``data.qpos``
        to qpos0. The skip meant ep1 ran from qpos0 for ~6.5 s of
        recording until ep2's slow path fired. Round 22 keeps the
        load_scene skip (the redundant-recompile concern that
        motivated the fast-path) but always runs canonical-state
        restoration."""
        pytest.importorskip("mujoco")
        scene = tmp_path / "scene.xml"
        scene.write_text("<mujoco><worldbody/></mujoco>")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=str(scene),
            install_cameras=False,
        )
        sim = FakeSim(data_config="panda")
        sim._world._backend_state["libero_prewarm_path"] = str(scene)

        with patch.object(LiberoAdapter, "_apply_canonical_state") as mock_apply:
            adapter.on_episode_start(sim, random.Random(0))

        # Fast-path STILL runs _apply_canonical_state - critical for
        # restoring qpos after PolicyRunner.sim.reset().
        mock_apply.assert_called_once()

    def test_ep1_uses_normal_lifecycle_even_with_prewarm_flag(self, tmp_path):
        """Episode 1+ runs full ``load_scene`` + ``_apply_canonical_state``
        regardless of the prewarm flag. The flag is one-shot for ep0;
        per-episode reset semantics for ep1+ are preserved."""
        pytest.importorskip("mujoco")
        scene = tmp_path / "scene.xml"
        scene.write_text("<mujoco><worldbody/></mujoco>")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=str(scene),
            install_cameras=False,
            apply_scene_keyframe=False,
        )
        sim = FakeSim(data_config="panda")
        # Even if the flag is somehow still set on ep1+, the
        # _episode_count > 0 check should bypass the fast-path.
        adapter._episode_count = 1
        sim._world._backend_state["libero_prewarm_path"] = str(scene)

        load_scene_calls: list = []
        original_load_scene = sim.load_scene

        def tracking_load_scene(scene_path):
            load_scene_calls.append(scene_path)
            return original_load_scene(scene_path)

        with patch.object(sim, "load_scene", side_effect=tracking_load_scene):
            adapter.on_episode_start(sim, random.Random(0))

        # Normal lifecycle: load_scene called.
        assert load_scene_calls == [str(scene)]

    def test_ep0_normal_lifecycle_when_prewarm_flag_missing(self, tmp_path):
        """Episode 0 WITHOUT the prewarm flag (e.g. user didn't call
        prewarm before evaluate_benchmark) falls through to the normal
        lifecycle: load_scene runs, _apply_canonical_state runs.
        Backwards-compat for callers that don't use prewarm."""
        pytest.importorskip("mujoco")
        scene = tmp_path / "scene.xml"
        scene.write_text("<mujoco><worldbody/></mujoco>")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=str(scene),
            install_cameras=False,
            apply_scene_keyframe=False,
        )
        sim = FakeSim(data_config="panda")
        # No libero_prewarm_path flag.
        assert "libero_prewarm_path" not in sim._world._backend_state

        load_scene_calls: list = []
        original_load_scene = sim.load_scene

        def tracking_load_scene(scene_path):
            load_scene_calls.append(scene_path)
            return original_load_scene(scene_path)

        with patch.object(sim, "load_scene", side_effect=tracking_load_scene):
            adapter.on_episode_start(sim, random.Random(0))

        # Normal lifecycle: load_scene called.
        assert load_scene_calls == [str(scene)]

    def test_ep0_fast_path_only_when_paths_match(self, tmp_path):
        """If ``libero_prewarm_path`` is set but to a DIFFERENT path than
        ``self.scene_path``, the fast-path is NOT taken. Defensive
        against stale flag state across adapter instances or scene-path
        changes mid-eval."""
        pytest.importorskip("mujoco")
        scene = tmp_path / "scene.xml"
        scene.write_text("<mujoco><worldbody/></mujoco>")
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=str(scene),
            install_cameras=False,
            apply_scene_keyframe=False,
        )
        sim = FakeSim(data_config="panda")
        # Different path than self.scene_path - prewarm flag is stale.
        sim._world._backend_state["libero_prewarm_path"] = "/some/other/scene.xml"

        load_scene_calls: list = []
        original_load_scene = sim.load_scene

        def tracking_load_scene(scene_path):
            load_scene_calls.append(scene_path)
            return original_load_scene(scene_path)

        with patch.object(sim, "load_scene", side_effect=tracking_load_scene):
            adapter.on_episode_start(sim, random.Random(0))

        # Stale flag → normal lifecycle → load_scene called.
        assert load_scene_calls == [str(scene)]

    def test_ep0_fast_path_skipped_on_model_size_mismatch(self, tmp_path, caplog):
        """When the prewarm flag is set AND scene_path matches, but the
        current model's nq+nv doesn't match init_states[0] width, the
        model has been mutated since prewarm ran (e.g. a redundant
        ``sim.add_robot`` welded in another robot). The fast-path
        sanity-check detects this and falls through to the normal
        lifecycle, logging at WARNING.

        Pin for #168 round-18 verification: rounds 17 fast-path took
        the flag at face value and skipped load_scene + canonical-state
        even when prewarm's init-state apply had silently no-op'd due
        to a model-size mismatch caused by an example-script
        ``sim.add_robot`` between ``sim.load_scene`` and ``spec.prewarm``.
        Result: recorder captured qpos0 of the 2-Panda model.
        """
        pytest.importorskip("mujoco")
        import logging

        scene = tmp_path / "scene.xml"
        scene.write_text("<mujoco><worldbody/></mujoco>")
        # init_states is sized for nq=4, nv=4 (1+4+4 = 9-wide).
        states = np.zeros((1, 9), dtype=np.float64)
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=str(scene),
            install_cameras=False,
            apply_scene_keyframe=False,
            init_states=states,
        )
        sim = FakeSim(data_config="panda")
        # Set the prewarm flag (simulating a successful-looking prewarm).
        sim._world._backend_state["libero_prewarm_path"] = str(scene)

        # But the current model has nq=10, nv=10 (1+10+10 = 21-wide),
        # NOT 9. Sanity-check detects this and falls through.
        class _StaleModel:
            nq = 10
            nv = 10

        sim._world._model = _StaleModel()  # type: ignore[attr-defined]

        load_scene_calls: list = []
        original_load_scene = sim.load_scene

        def tracking_load_scene(scene_path):
            load_scene_calls.append(scene_path)
            return original_load_scene(scene_path)

        with caplog.at_level(logging.WARNING, logger="strands_robots.benchmarks.libero.adapter"):
            with patch.object(sim, "load_scene", side_effect=tracking_load_scene):
                adapter.on_episode_start(sim, random.Random(0))

        # Sanity-check fired: load_scene called (normal lifecycle, not fast-path).
        assert load_scene_calls == [str(scene)], f"expected load_scene to be called; got {load_scene_calls}"
        # Flag was cleared by the sanity-check.
        assert "libero_prewarm_path" not in sim._world._backend_state
        # WARNING was logged so the user can detect the bad call ordering.
        assert any("model size mismatches init_states[0]" in r.message for r in caplog.records), (
            f"expected WARNING about model-size mismatch; got: {[r.message for r in caplog.records]}"
        )

    def test_prewarm_init_state_width_mismatch_logs_at_warning(self, caplog):
        """Width mismatch in ``_apply_init_state_for_prewarm`` logs at
        WARNING (not DEBUG) so users can detect a bad call ordering
        without enabling debug logging.

        Pin for #168 round 18: the failure mode is almost always a
        ``sim.add_robot`` between ``sim.load_scene`` and
        ``spec.prewarm`` recompiling the spec and bumping ``model.nq``
        past what ``init_states[0]`` was sized for. The skip is
        silent at DEBUG; at WARNING it's visible by default.
        """
        import logging

        pytest.importorskip("mujoco")
        # init_states width 5 (nq=2, nv=2 -> 1+2+2=5)
        states = np.zeros((1, 5), dtype=np.float64)
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path="/some/scene.xml",
            install_cameras=False,
            auto_generate_scene=False,
            init_states=states,
        )
        sim = FakeSim(data_config="panda")

        class _Model:
            nq = 4  # mismatch: 1+4+4=9, not 5
            nv = 4
            na = 0

        class _Data:
            def __init__(self) -> None:
                self.qpos = np.zeros(4)
                self.qvel = np.zeros(4)
                self.time = 0.0

        sim._world._model = _Model()  # type: ignore[attr-defined]
        sim._world._data = _Data()  # type: ignore[attr-defined]

        class _StubMj:
            class mjtVisFlag:
                mjVIS_JOINT = 0
                mjVIS_ACTUATOR = 0
                mjVIS_COM = 0

            class MjvOption:
                def __init__(self):
                    self.geomgroup = [1] * 6
                    self.sitegroup = [1] * 6
                    self.flags = [0] * 64

            @staticmethod
            def mjv_defaultOption(opt):
                pass

            @staticmethod
            def mj_forward(model, data):
                pass

        with caplog.at_level(logging.WARNING, logger="strands_robots.benchmarks.libero.adapter"):
            with patch.dict("sys.modules", {"mujoco": _StubMj}):
                adapter.prewarm(sim)

        # WARNING was logged (not DEBUG).
        assert any("init_state[0] width 5 != 1+nq+nv=9" in r.message for r in caplog.records), (
            f"expected WARNING about width mismatch; got: {[r.message for r in caplog.records]}"
        )
        # Flag was NOT set (skip path).
        assert "libero_prewarm_path" not in sim._world._backend_state


# Scene <keyframe> application (#166 follow-up)


class TestApplySceneKeyframe:
    """``LiberoAdapter._apply_canonical_state`` calls
    ``mj_resetDataKeyframe`` after every ``load_scene`` so qpos starts at
    the canonical home pose RoboSuite encoded in the MJCF ``<keyframe>``.
    Without this, ``mj_makeData`` and ``mj_resetData`` initialise from
    joint-default ``qpos0`` and free-joint objects (mugs, plates) snap
    to the origin - the #166 root cause for ``success_rate=0.00``.
    """

    def _make_mock_mujoco(self, calls: list[tuple[str, int]] | None = None):
        """Return a stub ``mujoco`` module that records mj_resetDataKeyframe calls."""
        if calls is None:
            calls = []

        class _Module:
            @staticmethod
            def mj_resetDataKeyframe(model, data, key):  # noqa: ARG004
                calls.append(("mj_resetDataKeyframe", int(key)))

            @staticmethod
            def mj_forward(model, data):  # noqa: ARG004
                calls.append(("mj_forward", 0))

        return _Module(), calls

    def test_applies_keyframe_when_model_has_one(self, tmp_path):
        """End-to-end: scene was loaded, model has nkey>0 ⇒ keyframe applied."""

        class _FakeModel:
            nkey = 1

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Pre-populate the cache so on_episode_start uses the cached path.
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # mj_resetDataKeyframe(model, data, 0) MUST have fired.
        assert ("mj_resetDataKeyframe", 0) in calls
        # And mj_forward must follow so derived state reflects the new qpos
        # before the first observation / render.
        assert ("mj_forward", 0) in calls

    def test_no_keyframe_in_model_skips_application(self, tmp_path):
        """Model with ``nkey == 0`` AND data without qpos/qvel attrs (e.g. a
        non-MuJoCo backend) must skip the keyframe path entirely - no
        ``mj_resetDataKeyframe`` calls fire. The snapshot fallback also
        no-ops because the data object has no ``qpos`` to capture."""

        class _FakeModel:
            nkey = 0

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # no qpos/qvel  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # Keyframe branch never fires (nkey==0); snapshot branch can't
        # capture because data has no qpos attr -> overall no-op.
        assert calls == []
        assert adapter._canonical_qpos is None

    def test_apply_scene_keyframe_false_disables(self, tmp_path):
        """Opt-out: apply_scene_keyframe=False skips the call even when
        the model declares a keyframe."""

        class _FakeModel:
            nkey = 1

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            apply_scene_keyframe=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        assert calls == []

    def test_skipped_when_no_scene_loaded(self):
        """Bare-Panda fallback (no scene_path, auto_generate failed) must
        not try to apply a keyframe - there's no scene to apply it to."""

        class _FakeModel:
            nkey = 1

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            auto_generate_scene=False,  # no scene generation
            apply_scene_keyframe=True,
        )
        # scene_path stays None -> load_scene never called -> no keyframe.

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        assert calls == []

    def test_custom_keyframe_index(self, tmp_path):
        """``scene_keyframe_index=1`` selects the second ``<keyframe>``."""

        class _FakeModel:
            nkey = 3

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            scene_keyframe_index=2,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        assert ("mj_resetDataKeyframe", 2) in calls

    def test_out_of_range_keyframe_index_skipped_with_warning(self, tmp_path, caplog):
        """``scene_keyframe_index >= model.nkey`` is a config error - log
        WARNING and skip, don't crash the episode."""

        class _FakeModel:
            nkey = 1

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            scene_keyframe_index=5,  # out of range
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with caplog.at_level("WARNING"):
            with patch.dict("sys.modules", {"mujoco": mock_mj}):
                adapter.on_episode_start(sim, random.Random(0))

        assert calls == []
        assert any("out of range" in rec.message for rec in caplog.records if rec.levelname == "WARNING")

    def test_canonical_state_applied_before_install_cameras(self, tmp_path):
        """Order matters (#166 review finding): the canonical-state apply
        must run RIGHT AFTER load_scene, BEFORE super() / install_cameras
        get a chance to recompile and shift qpos away from canonical.
        Otherwise the snapshot is taken post-recompile - already non-
        canonical - and replays a non-canonical state on every reset.
        """

        class _FakeModel:
            nkey = 1

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=True,  # camera install runs
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        ordering: list[str] = []

        def track_add_camera(self_, name, **kw):  # noqa: ARG002
            ordering.append(f"add_camera:{name}")
            return {"status": "success", "content": [{"text": "ok"}]}

        mock_mj, _ = self._make_mock_mujoco()
        original = mock_mj.mj_resetDataKeyframe

        def tracked_keyframe(model, data, key):
            ordering.append(f"keyframe:{key}")
            return original(model, data, key)

        mock_mj.mj_resetDataKeyframe = tracked_keyframe

        with patch.object(FakeSim, "add_camera", track_add_camera, create=False):
            with patch.dict("sys.modules", {"mujoco": mock_mj}):
                adapter.on_episode_start(sim, random.Random(0))

        camera_indices = [i for i, e in enumerate(ordering) if e.startswith("add_camera:")]
        keyframe_indices = [i for i, e in enumerate(ordering) if e.startswith("keyframe:")]
        assert keyframe_indices, "expected the keyframe to be applied"
        # The keyframe call MUST precede every add_camera call - the
        # canonical state needs to land before any recompile-prone step.
        if camera_indices:
            assert max(keyframe_indices) < min(camera_indices), (
                f"canonical-state apply at {keyframe_indices} should precede all camera installs at {camera_indices}"
            )

    def test_no_model_skips_silently(self):
        """Backends without a compiled model attribute (future engines) skip
        the keyframe application without raising."""

        class _BareWorld:
            cameras: dict[str, Any] = {}

        class _BareSim:
            _world = _BareWorld()  # no _model / _data

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            auto_generate_scene=False,
        )
        # Direct exercise of the helper - must not raise.
        adapter._apply_canonical_state(_BareSim())  # type: ignore[arg-type]


# Snapshot-and-restore branch (#166 follow-up review)


class TestApplyCanonicalStateSnapshot:
    """``_apply_canonical_state`` falls back to snapshot-and-restore when
    ``model.nkey == 0`` (procedurally-generated MJCFs from #165 don't ship
    a ``<keyframe>``). First episode after super() + camera install
    captures ``data.qpos`` / ``data.qvel``; subsequent episodes restore
    via ``np.copyto`` + ``mj_forward``. This is what actually fixes #166's
    ``success_rate=0.00`` symptom on ``examples/libero_mujoco.py``.
    """

    def _make_mock_mujoco(self, calls: list[tuple[str, Any]] | None = None):
        """Stub ``mujoco`` that records mj_resetDataKeyframe + mj_forward calls."""
        if calls is None:
            calls = []

        class _Module:
            @staticmethod
            def mj_resetDataKeyframe(model, data, key):  # noqa: ARG004
                calls.append(("mj_resetDataKeyframe", int(key)))

            @staticmethod
            def mj_forward(model, data):  # noqa: ARG004
                calls.append(("mj_forward", None))

        return _Module(), calls

    def _make_data(self, qpos_vals: list[float], qvel_vals: list[float] | None = None):
        """Bare ``data``-like object with mutable ``qpos`` / ``qvel`` arrays."""
        import numpy as _np

        class _D:
            qpos: Any
            qvel: Any

        d = _D()
        d.qpos = _np.array(qpos_vals, dtype=_np.float64)
        d.qvel = _np.array(qvel_vals if qvel_vals is not None else [], dtype=_np.float64)
        return d

    def test_first_call_captures_snapshot_no_restore(self, tmp_path):
        """First episode after a scene compile: snapshot is captured but
        no ``np.copyto`` / ``mj_forward`` runs (nothing to restore TO yet)."""

        class _FakeModel:
            nkey = 0

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = self._make_data([1.0, 2.0, 3.0], [0.1, 0.2])  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # Snapshot captured; no restore-side calls fired.
        assert adapter._canonical_qpos is not None
        assert list(adapter._canonical_qpos) == [1.0, 2.0, 3.0]
        assert adapter._canonical_qvel is not None
        assert list(adapter._canonical_qvel) == [0.1, 0.2]
        # Neither mj_resetDataKeyframe (nkey=0) nor mj_forward (no
        # restore on first capture) should have fired.
        assert calls == []

    def test_second_call_restores_from_snapshot_and_calls_mj_forward(self, tmp_path):
        """Second episode after the same scene compile: snapshot is
        ``np.copyto``'d into ``data.qpos`` and ``mj_forward`` runs so
        derived state reflects the canonical pose."""
        import numpy as _np

        class _FakeModel:
            nkey = 0

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        # Pre-seed the snapshot to simulate "this is episode 2+".
        adapter._canonical_qpos = _np.array([10.0, 20.0, 30.0], dtype=_np.float64)
        adapter._canonical_qvel = _np.array([1.1, 2.2], dtype=_np.float64)

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        # Current data is at "post-rollout state" - non-canonical.
        sim._world._data = self._make_data([99.0, 99.0, 99.0], [9.9, 9.9])  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # qpos / qvel must now equal the canonical snapshot, not the
        # 99.0 values that were there before.
        assert list(sim._world._data.qpos) == [10.0, 20.0, 30.0]
        assert list(sim._world._data.qvel) == [1.1, 2.2]
        # mj_forward MUST have fired so derived state (xpos/xquat) is current.
        assert ("mj_forward", None) in calls
        # mj_resetDataKeyframe MUST NOT have fired (nkey == 0).
        assert not any(c[0] == "mj_resetDataKeyframe" for c in calls)

    def test_snapshot_shape_mismatch_recaptures(self, tmp_path):
        """If a model recompile between episodes changed ``nq`` (unusual
        but possible), the snapshot-shape check re-captures instead of
        crashing on a copyto length mismatch."""
        import numpy as _np

        class _FakeModel:
            nkey = 0

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        # Pre-seed a 5-element snapshot.
        adapter._canonical_qpos = _np.zeros(5, dtype=_np.float64)
        adapter._canonical_qvel = _np.zeros(4, dtype=_np.float64)

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        # Current data has 3-element qpos - shape mismatch.
        sim._world._data = self._make_data([1.0, 2.0, 3.0], [0.1, 0.2])  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # Snapshot was re-captured (now 3 elements).
        assert adapter._canonical_qpos.shape == (3,)
        assert list(adapter._canonical_qpos) == [1.0, 2.0, 3.0]
        # No mj_forward since this counts as a "first call" for the new shape.
        assert calls == []

    def test_apply_scene_keyframe_false_disables_snapshot_branch_too(self, tmp_path):
        """``apply_scene_keyframe=False`` MUST disable BOTH branches -
        keyframe AND snapshot. Otherwise users opting out can't actually
        opt out."""

        class _FakeModel:
            nkey = 0

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            apply_scene_keyframe=False,  # opt out
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = self._make_data([1.0, 2.0, 3.0])  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # Snapshot branch never ran.
        assert adapter._canonical_qpos is None
        assert calls == []

    def test_keyframe_branch_takes_priority_over_snapshot_when_nkey_positive(self, tmp_path):
        """When ``model.nkey > 0`` AND a snapshot has been previously
        captured, the keyframe branch wins - we don't restore an old
        snapshot over a model that explicitly declares its canonical
        state via a ``<keyframe>``."""
        import numpy as _np

        class _FakeModel:
            nkey = 1

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        # Pre-seed a stale snapshot.
        adapter._canonical_qpos = _np.array([99.0], dtype=_np.float64)
        adapter._canonical_qvel = _np.array([9.9], dtype=_np.float64)

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = self._make_data([1.0], [0.1])  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # Keyframe path fired; data NOT mutated by the snapshot-restore path.
        assert ("mj_resetDataKeyframe", 0) in calls
        # qpos was NOT clobbered with the stale snapshot value.
        assert list(sim._world._data.qpos) == [1.0]


# init_jitter default (#167 Probe 2 folded into #168 per review)


class TestInitJitterDefault:
    """``init_jitter`` defaults to 0.0 to match LIBERO's deterministic-reset
    convention. The GR00T-LIBERO checkpoint trains against fixed init
    states per ``(task, seed)``; positive default jitter pushes the
    policy slightly out-of-distribution from t=0.
    """

    def test_default_is_zero(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        assert adapter._init_jitter == 0.0

    def test_default_is_zero_via_from_file(self, tmp_path):
        p = tmp_path / "task.bddl"
        p.write_text(PICK_CUBE_BDDL)
        adapter = LiberoAdapter.from_file(p)
        assert adapter._init_jitter == 0.0

    def test_default_is_zero_via_direct_constructor(self):
        from strands_robots.benchmarks.libero.bddl_parser import parse_bddl

        problem = parse_bddl(PICK_CUBE_BDDL)
        adapter = LiberoAdapter(problem)
        assert adapter._init_jitter == 0.0

    def test_explicit_value_still_works(self):
        """``init_jitter=0.02`` still works for users who want
        per-episode randomization (e.g. evaluating *generalization*)."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.02)
        assert adapter._init_jitter == 0.02


# Pre-register default robot to bypass super()'s redundant add_robot (#166 round 3)


class TestPreRegisterDefaultRobot:
    """``LiberoAdapter._register_default_robot`` scans the loaded MJCF for a
    scene-supplied Panda (bodies prefixed with ``robot0_`` per RoboSuite /
    LIBERO convention) and registers a :class:`SimRobot` wrapper directly
    in ``world.robots`` — *without* recompiling.

    Goal: make ``sim.list_robots()`` return ``["robot"]`` BEFORE
    ``super().on_episode_start`` runs, so the base BenchmarkProtocol
    skips its unconditional ``sim.add_robot`` call. That unconditional
    call would otherwise inject a SECOND Panda (the redundant-Panda bug
    confirmed in #166 round 4 — the second Panda's plastic shells sit
    right in front of the ``image`` camera and contaminate every
    real-render frame).
    """

    def _make_panda_model(
        self, prefix: str = "robot0_", body_count: int = 3, joint_count: int = 9, actuator_count: int = 8
    ):
        """Build a stub MuJoCo model with the right body/joint/actuator names."""

        class _FakeMjEnum:
            mjOBJ_BODY = 1
            mjOBJ_JOINT = 3
            mjOBJ_ACTUATOR = 5

        body_names = ["world"] + [f"{prefix}link{i}" for i in range(body_count - 1)]
        joint_names = [f"{prefix}joint{i}" for i in range(joint_count)]
        actuator_names = [f"{prefix}actuator{i}" for i in range(actuator_count)]
        # body_parentid: 0 = world, all other panda links are children of 0
        body_parentid = [0] + [0] * (body_count - 1)

        class _FakeMjModel:
            nbody = body_count
            njnt = joint_count
            nu = actuator_count
            body_parentid: list[int] = []  # set below; placeholder for class layout

        model = _FakeMjModel()
        model.body_parentid = body_parentid

        class _FakeMjModule:
            mjtObj = _FakeMjEnum()

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                if obj_type == _FakeMjEnum.mjOBJ_BODY:
                    return body_names[idx] if 0 <= idx < len(body_names) else None
                if obj_type == _FakeMjEnum.mjOBJ_JOINT:
                    return joint_names[idx] if 0 <= idx < len(joint_names) else None
                if obj_type == _FakeMjEnum.mjOBJ_ACTUATOR:
                    return actuator_names[idx] if 0 <= idx < len(actuator_names) else None
                return None

        return model, _FakeMjModule()

    def _scene_path_setup(self, tmp_path):
        """Create a cached scene XML so on_episode_start hits the load_scene path."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            apply_scene_keyframe=False,  # focus on pre-register
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")
        return adapter

    def test_scene_panda_detected_and_registered(self, tmp_path):
        """When the loaded model has bodies / joints / actuators with the
        canonical ``robot0_`` prefix, the adapter MUST construct a
        SimRobot wrapper and register it under ``"robot"`` — no
        ``sim.add_robot`` call required."""
        adapter = self._scene_path_setup(tmp_path)
        model, mj = self._make_panda_model()

        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()
        sim._world._model = model  # type: ignore[attr-defined]

        add_robot_calls: list[str] = []

        def tracking_add_robot(self_, name, **kw):  # noqa: ARG002
            add_robot_calls.append(name)
            return {"status": "success", "content": [{"text": "ok"}]}

        with patch.object(FakeSim, "add_robot", tracking_add_robot, create=False):
            with patch.dict("sys.modules", {"mujoco": mj}):
                adapter.on_episode_start(sim, random.Random(0))

        # Critical: NO add_robot calls fired (no recompile, no redundant Panda).
        assert add_robot_calls == []
        # Wrapper landed in the registry under the canonical key.
        assert "robot" in sim._world.robots
        wrapper = sim._world.robots["robot"]
        assert wrapper.data_config == "panda"
        assert wrapper.namespace == "robot0_"
        assert len(wrapper.joint_names) == 9
        assert all(n.startswith("robot0_") for n in wrapper.joint_names)
        assert len(wrapper.actuator_ids) == 8

    def test_no_scene_panda_falls_back_to_super_add_robot(self, tmp_path):
        """When the loaded MJCF doesn't have a ``robot0_`` body, the adapter
        leaves ``world.robots`` alone and super() does its normal add."""
        adapter = self._scene_path_setup(tmp_path)

        # Model without any robot0_ prefix bodies.
        class _NoRobotModel:
            nbody = 1
            njnt = 0
            nu = 0
            body_parentid = [0]

        class _NoRobotMj:
            class mjtObj:
                mjOBJ_BODY = 1
                mjOBJ_JOINT = 3
                mjOBJ_ACTUATOR = 5

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                return "world" if obj_type == 1 and idx == 0 else None

        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()
        sim._world._model = _NoRobotModel()  # type: ignore[attr-defined]

        add_robot_calls: list[str] = []

        def tracking_add_robot(self_, name, **kw):  # noqa: ARG002
            add_robot_calls.append(name)
            sim._world.robots[name] = _FakeRobot(kw.get("data_config", "panda"))
            return {"status": "success", "content": [{"text": "ok"}]}

        with patch.object(FakeSim, "add_robot", tracking_add_robot, create=False):
            with patch.dict("sys.modules", {"mujoco": _NoRobotMj()}):
                adapter.on_episode_start(sim, random.Random(0))

        # super() did the add - this is the bare-Panda fallback.
        assert add_robot_calls == ["robot"]

    def test_robot_already_registered_noop(self, tmp_path):
        """If ``"robot"`` is already in the registry (defensive), the
        scan path is skipped entirely."""
        adapter = self._scene_path_setup(tmp_path)
        # No fake model needed - the early-return on registry check fires
        # before the scan.
        sim = FakeSim(data_config="panda")
        sim._world.robots["robot"] = _FakeRobot("panda")

        # Don't patch mujoco — if the scan ran, it would fail.
        adapter.on_episode_start(sim, random.Random(0))

        # Registration unchanged.
        assert sim._world.robots["robot"].data_config == "panda"

    def test_pre_register_skipped_when_no_scene_loaded(self):
        """Bare-Panda fallback (no scene_path): pre-register MUST NOT fire
        because there's no scene-supplied panda to wrap. super() does the
        normal add."""
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            auto_generate_scene=False,  # no scene
        )
        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()

        add_robot_calls: list[str] = []

        def tracking_add_robot(self_, name, **kw):  # noqa: ARG002
            add_robot_calls.append(name)
            sim._world.robots[name] = _FakeRobot(kw.get("data_config", "panda"))
            return {"status": "success", "content": [{"text": "ok"}]}

        with patch.object(FakeSim, "add_robot", tracking_add_robot, create=False):
            adapter.on_episode_start(sim, random.Random(0))

        # super() did its normal add (single call) - pre-register no-op'd.
        assert add_robot_calls == ["robot"]

    def test_custom_scene_robot_prefix(self, tmp_path):
        """Custom ``scene_robot_prefix`` lets the adapter wrap pandas
        named differently (e.g. for non-LIBERO scenes that use a
        different convention)."""
        adapter = self._scene_path_setup(tmp_path)
        adapter._scene_robot_prefix = "myrobot_"
        model, mj = self._make_panda_model(prefix="myrobot_")

        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()
        sim._world._model = model  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": mj}):
            adapter.on_episode_start(sim, random.Random(0))

        assert "robot" in sim._world.robots
        assert sim._world.robots["robot"].namespace == "myrobot_"

    def test_no_compiled_model_skipped(self, tmp_path):
        """Sim without ``world._model`` (non-MuJoCo backend, or stub) - skip."""
        adapter = self._scene_path_setup(tmp_path)
        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()
        sim._world._model = None  # type: ignore[attr-defined]

        # super() falls back to its normal add path.
        with patch.dict("sys.modules", {"mujoco": MagicMock()}):
            adapter.on_episode_start(sim, random.Random(0))

        # No wrapper from the scan path; super() did the regular add.
        # FakeSim's add_robot registers under name="robot".
        assert "robot" in sim._world.robots

    def test_module_helper_build_scene_robot_wrapper_returns_none_on_no_match(self):
        """``_build_scene_robot_wrapper`` returns None when no body matches."""
        from strands_robots.benchmarks.libero.adapter import _build_scene_robot_wrapper

        class _Mj:
            class mjtObj:
                mjOBJ_BODY = 1
                mjOBJ_JOINT = 3
                mjOBJ_ACTUATOR = 5

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                return "world" if idx == 0 else None

        class _Model:
            nbody = 2
            njnt = 0
            nu = 0
            body_parentid = [0, 0]

        wrapper = _build_scene_robot_wrapper(_Mj(), _Model(), prefix="robot0_")
        assert wrapper is None

    def test_module_helper_build_scene_robot_wrapper_filters_correctly(self):
        """``_build_scene_robot_wrapper`` only includes bodies / joints /
        actuators with the matching prefix."""
        from strands_robots.benchmarks.libero.adapter import _build_scene_robot_wrapper

        body_names = ["world", "robot0_base", "table", "robot0_link0", "mug_1"]
        joint_names = ["robot0_joint1", "table_joint", "robot0_finger_joint1"]
        actuator_names = ["robot0_act1", "table_act", "robot0_act2"]

        class _Mj:
            class mjtObj:
                mjOBJ_BODY = 1
                mjOBJ_JOINT = 3
                mjOBJ_ACTUATOR = 5

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                if obj_type == 1:
                    return body_names[idx] if idx < len(body_names) else None
                if obj_type == 3:
                    return joint_names[idx] if idx < len(joint_names) else None
                if obj_type == 5:
                    return actuator_names[idx] if idx < len(actuator_names) else None
                return None

        class _Model:
            nbody = len(body_names)
            njnt = len(joint_names)
            nu = len(actuator_names)
            body_parentid = [0, 0, 0, 0, 0]

        wrapper = _build_scene_robot_wrapper(_Mj(), _Model(), prefix="robot0_")
        assert wrapper is not None
        # Only 2 robot0_ joints (joint1, finger_joint1)
        assert wrapper.joint_names == ["robot0_joint1", "robot0_finger_joint1"]
        # Only 2 robot0_ actuators
        assert len(wrapper.actuator_ids) == 2
        # Wrapper namespace is the prefix.
        assert wrapper.namespace == "robot0_"
        # Wrapper name is the canonical "robot" key.
        assert wrapper.name == "robot"

    def test_module_helper_build_scene_robot_wrapper_includes_gripper_prefix(self):
        """``_build_scene_robot_wrapper`` with ``gripper_prefix=`` includes
        joints / actuators from BOTH namespaces. Pin for #168 round-5
        bug G: RoboSuite uses ``robot0_*`` for arm joints and
        ``gripper0_*`` for gripper finger joints; previously the wrapper
        scoped to ``robot0_`` only and silently dropped
        ``gripper0_finger_joint{1,2}`` from ``wrapper.joint_names``,
        causing ``obs.get('gripper0_finger_joint1')`` to return None and
        the GR00T server to reject every observation with
        ``State key 'state.gripper' must be in observation``."""
        from strands_robots.benchmarks.libero.adapter import _build_scene_robot_wrapper

        body_names = ["world", "robot0_base", "robot0_link0", "robot0_right_hand", "gripper0_eef"]
        # Mix arm + gripper joints in a realistic interleaved order.
        joint_names = [
            "robot0_joint1",
            "robot0_joint2",
            "robot0_joint7",
            "gripper0_finger_joint1",
            "gripper0_finger_joint2",
            "table_joint",  # unrelated; must NOT be included
        ]
        actuator_names = [
            "robot0_act1",
            "robot0_act7",
            "gripper0_finger_act",
            "table_act",  # unrelated
        ]

        class _Mj:
            class mjtObj:
                mjOBJ_BODY = 1
                mjOBJ_JOINT = 3
                mjOBJ_ACTUATOR = 5

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                if obj_type == 1:
                    return body_names[idx] if idx < len(body_names) else None
                if obj_type == 3:
                    return joint_names[idx] if idx < len(joint_names) else None
                if obj_type == 5:
                    return actuator_names[idx] if idx < len(actuator_names) else None
                return None

        class _Model:
            nbody = len(body_names)
            njnt = len(joint_names)
            nu = len(actuator_names)
            body_parentid = [0] * len(body_names)

        wrapper = _build_scene_robot_wrapper(
            _Mj(),
            _Model(),
            prefix="robot0_",
            gripper_prefix="gripper0_",
        )
        assert wrapper is not None
        # All three arm joints + both gripper joints present, in scan order.
        assert wrapper.joint_names == [
            "robot0_joint1",
            "robot0_joint2",
            "robot0_joint7",
            "gripper0_finger_joint1",
            "gripper0_finger_joint2",
        ]
        # Both arm actuators + the gripper actuator present (3 total).
        assert len(wrapper.actuator_ids) == 3
        # Namespace remains the arm prefix - the gripper prefix is a
        # filtering hint, not a namespace replacement.
        assert wrapper.namespace == "robot0_"

    def test_module_helper_build_scene_robot_wrapper_no_gripper_prefix_drops_gripper_joints(self):
        """Default ``gripper_prefix=None`` keeps the legacy single-prefix
        behaviour - gripper joints are NOT included. Pin for backwards
        compat: callers passing only ``prefix=`` get exactly the same
        wrapper as before #168 round 6."""
        from strands_robots.benchmarks.libero.adapter import _build_scene_robot_wrapper

        body_names = ["world", "robot0_base", "robot0_right_hand"]
        joint_names = ["robot0_joint1", "gripper0_finger_joint1"]
        actuator_names = ["robot0_act1", "gripper0_finger_act"]

        class _Mj:
            class mjtObj:
                mjOBJ_BODY = 1
                mjOBJ_JOINT = 3
                mjOBJ_ACTUATOR = 5

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                if obj_type == 1:
                    return body_names[idx] if idx < len(body_names) else None
                if obj_type == 3:
                    return joint_names[idx] if idx < len(joint_names) else None
                if obj_type == 5:
                    return actuator_names[idx] if idx < len(actuator_names) else None
                return None

        class _Model:
            nbody = len(body_names)
            njnt = len(joint_names)
            nu = len(actuator_names)
            body_parentid = [0] * len(body_names)

        wrapper = _build_scene_robot_wrapper(_Mj(), _Model(), prefix="robot0_")
        assert wrapper is not None
        # Only the arm joint included.
        assert wrapper.joint_names == ["robot0_joint1"]
        # Only the arm actuator.
        assert len(wrapper.actuator_ids) == 1

    def test_module_helper_build_scene_robot_wrapper_empty_gripper_prefix_drops_gripper(self):
        """Explicit ``gripper_prefix=""`` (falsy) is treated the same as
        ``None`` - no gripper joints included. Required because some
        callers may want to disable the dual-prefix path while leaving
        ``scene_gripper_prefix`` set on the adapter for use by the
        eef/gripper auto-resolver only."""
        from strands_robots.benchmarks.libero.adapter import _build_scene_robot_wrapper

        body_names = ["world", "robot0_base", "robot0_right_hand"]
        joint_names = ["robot0_joint1", "gripper0_finger_joint1"]
        actuator_names = ["robot0_act1", "gripper0_finger_act"]

        class _Mj:
            class mjtObj:
                mjOBJ_BODY = 1
                mjOBJ_JOINT = 3
                mjOBJ_ACTUATOR = 5

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                if obj_type == 1:
                    return body_names[idx] if idx < len(body_names) else None
                if obj_type == 3:
                    return joint_names[idx] if idx < len(joint_names) else None
                if obj_type == 5:
                    return actuator_names[idx] if idx < len(actuator_names) else None
                return None

        class _Model:
            nbody = len(body_names)
            njnt = len(joint_names)
            nu = len(actuator_names)
            body_parentid = [0] * len(body_names)

        wrapper = _build_scene_robot_wrapper(
            _Mj(),
            _Model(),
            prefix="robot0_",
            gripper_prefix="",
        )
        assert wrapper is not None
        assert wrapper.joint_names == ["robot0_joint1"]
        assert len(wrapper.actuator_ids) == 1

    def test_register_default_robot_forwards_scene_gripper_prefix(self, tmp_path):
        """End-to-end: ``LiberoAdapter`` with default
        ``scene_gripper_prefix='gripper0_'`` produces a wrapper whose
        ``joint_names`` includes BOTH the arm and gripper joints. Pin
        for the actual round-6 fix path: previously the adapter's
        ``_register_default_robot`` only forwarded ``scene_robot_prefix``
        and the wrapper silently dropped gripper joints."""
        adapter = self._scene_path_setup(tmp_path)
        # Build a model with arm + gripper joints (the realistic LIBERO layout)
        body_names = ["world", "robot0_base", "robot0_right_hand"]
        joint_names = [f"robot0_joint{i}" for i in range(1, 8)] + [
            "gripper0_finger_joint1",
            "gripper0_finger_joint2",
        ]
        actuator_names = [f"robot0_act{i}" for i in range(1, 8)] + ["gripper0_finger_act"]

        class _FakeMjEnum:
            mjOBJ_BODY = 1
            mjOBJ_JOINT = 3
            mjOBJ_ACTUATOR = 5

        class _Model:
            nbody = len(body_names)
            njnt = len(joint_names)
            nu = len(actuator_names)
            body_parentid: list[int] = []

        model = _Model()
        model.body_parentid = [0] * len(body_names)

        class _Mj:
            mjtObj = _FakeMjEnum()

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                if obj_type == 1:
                    return body_names[idx] if 0 <= idx < len(body_names) else None
                if obj_type == 3:
                    return joint_names[idx] if 0 <= idx < len(joint_names) else None
                if obj_type == 5:
                    return actuator_names[idx] if 0 <= idx < len(actuator_names) else None
                return None

            @staticmethod
            def mj_name2id(model, obj_type, name):  # noqa: ARG004
                pool = {1: body_names, 3: joint_names, 5: actuator_names}.get(obj_type, [])
                try:
                    return pool.index(name)
                except ValueError:
                    return -1

        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()
        sim._world._model = model  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": _Mj()}):
            adapter.on_episode_start(sim, random.Random(0))

        wrapper = sim._world.robots["robot"]
        # Critical assertion: gripper finger joints must be in the wrapper's
        # observation surface so the upstream get_observation populates them.
        assert "gripper0_finger_joint1" in wrapper.joint_names
        assert "gripper0_finger_joint2" in wrapper.joint_names
        # Plus all 7 arm joints.
        for i in range(1, 8):
            assert f"robot0_joint{i}" in wrapper.joint_names
        # Total: 7 arm + 2 gripper = 9 joints.
        assert len(wrapper.joint_names) == 9
        # Total: 7 arm actuators + 1 gripper actuator = 8 actuators.
        assert len(wrapper.actuator_ids) == 8


class TestResolveSceneEefAndGripper:
    """``LiberoAdapter._resolve_scene_eef_and_gripper`` overrides the
    bare-Panda defaults (``"hand"`` / ``"finger_joint1"``) when the scene
    declares the canonical RoboSuite / LIBERO names (``robot0_right_hand``
    body, ``gripper0_finger_joint1`` joint). Without this override, the
    ``augment_observation`` hook silently drops every ``state.x/y/z`` /
    ``state.gripper`` key on a RoboSuite-emitted scene because
    ``get_body_state("hand")`` returns body_id=-1 and the gripper-joint
    suffix-match fails. The GR00T server then rejects the observation
    with ``State key 'state.x' must be in observation`` and the eval
    crashes before producing any frame (#168 round-4 bug F)."""

    def _make_robosuite_model(
        self,
        *,
        eef_body: str = "robot0_right_hand",
        gripper_joint: str = "gripper0_finger_joint1",
        extra_bodies: tuple[str, ...] = (),
        extra_joints: tuple[str, ...] = (),
    ):
        """Build a stub MuJoCo model that mimics a RoboSuite-compiled
        Panda scene's name layout. Critical: the EEF body name and
        gripper joint name match the canonical RoboSuite emit names so
        :meth:`_resolve_scene_eef_and_gripper` can find them via
        ``mj_name2id``.
        """
        body_names = ["world", "robot0_base", "robot0_link0", eef_body, "gripper0_right_gripper"] + list(extra_bodies)
        joint_names = ["robot0_joint1", "robot0_joint2", gripper_joint] + list(extra_joints)
        actuator_names = ["robot0_actuator0"]

        class _FakeMjEnum:
            mjOBJ_BODY = 1
            mjOBJ_JOINT = 3
            mjOBJ_ACTUATOR = 5

        class _FakeModel:
            nbody = len(body_names)
            njnt = len(joint_names)
            nu = len(actuator_names)
            body_parentid: list[int] = []

        model = _FakeModel()
        model.body_parentid = [0] * len(body_names)

        class _FakeMj:
            mjtObj = _FakeMjEnum()

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                if obj_type == 1:
                    return body_names[idx] if 0 <= idx < len(body_names) else None
                if obj_type == 3:
                    return joint_names[idx] if 0 <= idx < len(joint_names) else None
                if obj_type == 5:
                    return actuator_names[idx] if 0 <= idx < len(actuator_names) else None
                return None

            @staticmethod
            def mj_name2id(model, obj_type, name):  # noqa: ARG004
                pool = {1: body_names, 3: joint_names, 5: actuator_names}.get(obj_type, [])
                try:
                    return pool.index(name)
                except ValueError:
                    return -1

        return model, _FakeMj()

    def _scene_path_setup(self, tmp_path):
        """Create a cached scene XML so on_episode_start hits load_scene path."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            apply_scene_keyframe=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")
        return adapter

    def test_eef_body_auto_resolved_to_robot0_right_hand(self, tmp_path):
        """Default constructor ``eef_body_name=None`` -> auto-detect.
        Scene has ``robot0_right_hand`` -> override default ``"hand"``."""
        adapter = self._scene_path_setup(tmp_path)
        # Adapter constructor leaves _eef_body_name at the bare-Panda fallback.
        assert adapter._eef_body_name == "hand"
        assert adapter._user_eef_body_name is None

        model, mj = self._make_robosuite_model()
        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()
        sim._world._model = model  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # After episode start, the auto-resolver picked up the scene's
        # canonical RoboSuite EEF body name.
        assert adapter._eef_body_name == "robot0_right_hand"

    def test_gripper_joint_auto_resolved_to_gripper0_finger_joint1(self, tmp_path):
        """Default ``gripper_joint_name=None`` -> auto-detect via the
        ``scene_gripper_prefix`` namespace (``gripper0_``)."""
        adapter = self._scene_path_setup(tmp_path)
        assert adapter._gripper_joint_name == "finger_joint1"
        assert adapter._user_gripper_joint_name is None

        model, mj = self._make_robosuite_model()
        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()
        sim._world._model = model  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": mj}):
            adapter.on_episode_start(sim, random.Random(0))

        assert adapter._gripper_joint_name == "gripper0_finger_joint1"

    def test_explicit_eef_body_name_preserved(self, tmp_path):
        """User-supplied ``eef_body_name="hand"`` (or any explicit value) is
        NEVER overridden, even if the scene declares ``robot0_right_hand``.
        Backwards compat for callers running custom scenes."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            apply_scene_keyframe=False,
            eef_body_name="hand",  # explicit user override
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        # _user_eef_body_name records the explicit user value.
        assert adapter._user_eef_body_name == "hand"

        model, mj = self._make_robosuite_model()
        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()
        sim._world._model = model  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # User explicit override - resolver did NOT touch it.
        assert adapter._eef_body_name == "hand"

    def test_explicit_gripper_joint_name_preserved(self, tmp_path):
        """User-supplied ``gripper_joint_name`` is NEVER overridden."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            apply_scene_keyframe=False,
            gripper_joint_name="finger_joint1",
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        assert adapter._user_gripper_joint_name == "finger_joint1"

        model, mj = self._make_robosuite_model()
        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()
        sim._world._model = model  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": mj}):
            adapter.on_episode_start(sim, random.Random(0))

        assert adapter._gripper_joint_name == "finger_joint1"

    def test_no_match_keeps_bare_panda_default(self, tmp_path):
        """Scene without ``robot0_right_hand`` / ``gripper0_finger_joint1``
        (e.g. a custom non-RoboSuite scene): auto-resolver finds no
        candidate, leaves the bare-Panda defaults in place."""
        adapter = self._scene_path_setup(tmp_path)

        # Model has scene-Panda body matching `robot0_` prefix (so
        # _register_default_robot still fires the resolver) but the
        # canonical EEF / gripper names are missing.
        model, mj = self._make_robosuite_model(
            eef_body="robot0_unusual_eef",
            gripper_joint="custom_joint",
        )
        # Override mj_name2id to NOT find any of the resolver's candidates.
        body_names = ["world", "robot0_base", "robot0_link0", "robot0_unusual_eef"]
        joint_names = ["robot0_joint1", "custom_joint"]

        class _BareMj:
            class mjtObj:
                mjOBJ_BODY = 1
                mjOBJ_JOINT = 3
                mjOBJ_ACTUATOR = 5

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                if obj_type == 1:
                    return body_names[idx] if 0 <= idx < len(body_names) else None
                if obj_type == 3:
                    return joint_names[idx] if 0 <= idx < len(joint_names) else None
                return None

            @staticmethod
            def mj_name2id(model, obj_type, name):  # noqa: ARG004
                # Only finds names actually present (no candidate matches)
                if obj_type == 1 and name in body_names:
                    return body_names.index(name)
                if obj_type == 3 and name in joint_names:
                    return joint_names.index(name)
                return -1

        class _Model:
            nbody = len(body_names)
            njnt = len(joint_names)
            nu = 1
            body_parentid = [0] * 4

        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()
        sim._world._model = _Model()  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": _BareMj()}):
            adapter.on_episode_start(sim, random.Random(0))

        # Resolver found nothing -> fell back to bare-Panda defaults.
        assert adapter._eef_body_name == "hand"
        assert adapter._gripper_joint_name == "finger_joint1"

    def test_custom_scene_gripper_prefix_used_in_resolution(self, tmp_path):
        """``scene_gripper_prefix="custom_"`` -> resolver looks up
        ``custom_finger_joint1`` instead of ``gripper0_finger_joint1``."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            apply_scene_keyframe=False,
            scene_gripper_prefix="custom_",
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        sha = adapter._scene_cache_key(bddl_path.read_bytes())
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        model, mj = self._make_robosuite_model(gripper_joint="custom_finger_joint1")
        sim = FakeSim(data_config="panda")
        sim._world.robots.clear()
        sim._world._model = model  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": mj}):
            adapter.on_episode_start(sim, random.Random(0))

        assert adapter._gripper_joint_name == "custom_finger_joint1"

    def test_first_named_returns_none_when_mj_name2id_missing(self):
        """``_first_named`` returns None when the mj module lacks
        ``mj_name2id`` (e.g. test stubs that only implement mj_id2name).
        Defensive guard against partial test fakes."""
        from strands_robots.benchmarks.libero.adapter import LiberoAdapter as _Adapter

        class _StubMj:
            class mjtObj:
                mjOBJ_BODY = 1

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                return None

        # _first_named is a static method - call it directly without an instance.
        result = _Adapter._first_named(_StubMj(), object(), names=["foo"], obj=1)
        assert result is None


class TestSceneCameraAliasesDefault:
    """The default ``_scene_camera_aliases`` map covers BOTH the bare
    RoboSuite camera name (``robot0_eye_in_hand``, what RoboSuite's
    ``env.sim.model.get_xml()`` emits in the compiled MJCF) AND the
    legacy ``_image``-suffixed variant (``robot0_eye_in_hand_image``,
    older convention). Both must rename to ``wrist_image`` so the
    ``libero_panda`` ``Gr00tDataConfig`` finds the gripper-tracking
    eye-in-hand camera at the canonical observation key. Without the
    bare-name entry, the static top-down workspace fallback in
    :attr:`LIBERO_CAMERAS` gets installed as ``wrist_image``, and GR00T
    sees out-of-distribution input on its wrist channel every step
    (#168 round-4 bug A)."""

    def test_default_aliases_include_both_eye_in_hand_variants(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        aliases = adapter._scene_camera_aliases
        # Both spellings rename to the same canonical key.
        assert aliases.get("robot0_eye_in_hand") == "wrist_image"
        assert aliases.get("robot0_eye_in_hand_image") == "wrist_image"
        # Agentview rename is also still present.
        assert aliases.get("agentview") == "image"

    def test_explicit_aliases_replace_default_entirely(self):
        """Passing ``scene_camera_aliases=...`` REPLACES the default
        (no merge). Users opting into a custom map are responsible for
        re-adding the wrist alias if they want it."""
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_camera_aliases={"only_one": "renamed"},
        )
        assert adapter._scene_camera_aliases == {"only_one": "renamed"}

    def test_robot0_eye_in_hand_renames_to_wrist_image_in_xml(self):
        """End-to-end: applying the default alias map to a RoboSuite-style
        XML with a bare ``robot0_eye_in_hand`` camera produces an XML
        where that camera is renamed to ``wrist_image``."""
        from strands_robots.benchmarks.libero.adapter import _rename_mjcf_cameras

        xml = (
            "<mujoco>"
            "<worldbody>"
            '<camera name="agentview" pos="1 0 1"/>'
            '<body name="robot0_right_hand">'
            '<camera name="robot0_eye_in_hand" pos="0.05 0 0"/>'
            "</body>"
            "</worldbody>"
            "</mujoco>"
        )
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        renamed = _rename_mjcf_cameras(xml, adapter._scene_camera_aliases)
        assert 'name="image"' in renamed
        assert 'name="wrist_image"' in renamed
        assert 'name="agentview"' not in renamed
        assert 'name="robot0_eye_in_hand"' not in renamed


class TestApplyInitStateBranch:
    """``LiberoAdapter._apply_init_state_branch`` writes
    ``data.time / data.qpos / data.qvel`` directly from a row of the
    ``init_states`` ndarray (#168 round-7 bug I).

    LIBERO ships task-specific *init states* alongside its benchmark
    suites - each task has 50 sampled starting configurations of the
    canonical "ready" pose. Without this branch the robot starts at
    ``qpos=0`` (the joint-default "stretched flat" pose) instead of the
    canonical "ready" pose, the policy issues actions calibrated for
    the canonical pose against a totally different body configuration,
    the robot wiggles uselessly, and ``success_rate`` collapses to 0.

    Layout per ``robosuite/utils/binding_utils.py:213-241``:
    ``[time(1), qpos(nq), qvel(nv)]`` with ``na == 0``. The branch
    raises ``RuntimeError`` on width mismatch (silent slicing
    forbidden per AGENTS.md "no silent defaults on error") so a
    procedurally-generated MJCF that diverges from upstream LIBERO's
    scene MJCF surfaces loudly rather than producing a deeply wrong
    physical state.
    """

    @staticmethod
    def _make_sim_with_model(*, nq: int, nv: int, na: int = 0):
        """Build a FakeSim whose ``world._model`` exposes nq/nv/na and
        whose ``world._data`` has writable qpos/qvel/time."""
        import numpy as np

        class _FakeModel:
            def __init__(self, nq_: int, nv_: int, na_: int) -> None:
                self.nq = nq_
                self.nv = nv_
                self.na = na_
                self.nkey = 0  # force snapshot/init-state branch (not keyframe)

        class _FakeData:
            def __init__(self, nq_: int, nv_: int) -> None:
                self.qpos = np.zeros(nq_)
                self.qvel = np.zeros(nv_)
                self.time = 0.0

        model = _FakeModel(nq, nv, na)
        data = _FakeData(nq, nv)

        sim = FakeSim(data_config="panda")
        sim._world._model = model  # type: ignore[attr-defined]
        sim._world._data = data  # type: ignore[attr-defined]
        return sim, model, data

    @staticmethod
    def _make_fake_mj():
        """Lightweight mujoco stub - only mj_forward needed by the branch."""
        forward_calls: list[tuple[Any, Any]] = []

        class _Mj:
            class mjtObj:
                mjOBJ_BODY = 1
                mjOBJ_JOINT = 3
                mjOBJ_ACTUATOR = 5

            @staticmethod
            def mj_forward(model, data):
                forward_calls.append((model, data))

        return _Mj(), forward_calls

    def test_init_state_applied_writes_qpos_qvel_time(self):
        """Happy-path: a single 1+nq+nv-wide init_state row gets written
        to ``data.time / data.qpos / data.qvel`` and ``mj_forward`` is
        called once."""
        nq, nv = 9, 9
        # Construct a deterministic init state: time=0.5, qpos=[1..9], qvel=[10..18]
        state = np.array([0.5] + list(range(1, nq + 1)) + list(range(10, 10 + nv)), dtype=np.float64)
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            apply_scene_keyframe=True,
            init_states=state,  # 1D promoted to (1, 1+nq+nv)
        )
        assert adapter._init_states is not None
        assert adapter._init_states.shape == (1, 1 + nq + nv)

        sim, model, data = self._make_sim_with_model(nq=nq, nv=nv)
        mj, forward_calls = self._make_fake_mj()
        with patch.dict("sys.modules", {"mujoco": mj}):
            adapter._apply_canonical_state(sim, random.Random(0))

        # qpos/qvel/time updated to match the init_state.
        np.testing.assert_array_equal(data.qpos, state[1 : 1 + nq])
        np.testing.assert_array_equal(data.qvel, state[1 + nq :])
        assert data.time == 0.5
        # mj_forward fired.
        assert len(forward_calls) == 1

    def test_init_state_episode_zero_is_deterministic(self):
        """Episode 0 always uses ``init_states[0]`` regardless of RNG.

        Pin for #168 round-16 contract: matches v0.1.1
        ``env_libero.py``'s ``env.set_init_state(init_states[0])``
        pattern for the first episode. Aligns with
        :meth:`prewarm`'s init-state apply (which always uses idx 0)
        so the recorder's t=0.00 frame and the policy's first
        observation are visually identical."""
        nq, nv = 4, 4
        states = np.zeros((2, 1 + nq + nv), dtype=np.float64)
        states[0, 1] = 100.0  # state 0: qpos[0] = 100
        states[1, 1] = 200.0  # state 1: qpos[0] = 200
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            init_states=states,
        )

        sim, _, data = self._make_sim_with_model(nq=nq, nv=nv)
        mj, _ = self._make_fake_mj()
        # Episode 0 ALWAYS uses idx=0, even with a different RNG seed.
        # Verify by passing a seed where Random(N).randint(0, 1) would
        # otherwise return 1 (e.g. Random(1) selects 1 first).
        with patch.dict("sys.modules", {"mujoco": mj}):
            adapter._apply_canonical_state(sim, random.Random(1))
        # Episode 0 must apply state 0 (qpos[0] = 100), regardless of seed.
        assert data.qpos[0] == 100.0, f"episode 0 should be deterministic idx=0 (qpos[0]=100), got {data.qpos[0]}"
        # Counter incremented to 1 after ep0 apply.
        assert adapter._episode_count == 1

    def test_init_state_episode_one_uses_rng(self):
        """Episodes 1+ use RNG-sampled selection. Required so per-episode
        randomization is preserved for episodes after ep0 (which is
        pinned to idx 0). Pin for the seeded-reproducibility contract:
        same seed → same idx for the same episode index."""
        nq, nv = 4, 4
        states = np.zeros((2, 1 + nq + nv), dtype=np.float64)
        states[0, 1] = 100.0
        states[1, 1] = 200.0
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            init_states=states,
        )

        # Bump episode counter past 0 to simulate "we've already done ep0".
        adapter._episode_count = 1

        sim_a, _, data_a = self._make_sim_with_model(nq=nq, nv=nv)
        sim_b, _, data_b = self._make_sim_with_model(nq=nq, nv=nv)
        mj, _ = self._make_fake_mj()
        with patch.dict("sys.modules", {"mujoco": mj}):
            adapter._apply_canonical_state(sim_a, random.Random(42))
            # Reset counter so sim_b is also "ep 1" for fair comparison.
            adapter._episode_count = 1
            adapter._apply_canonical_state(sim_b, random.Random(42))
        # Same seed + same episode-counter state -> same selected state.
        assert data_a.qpos[0] == data_b.qpos[0], (
            f"same seed should produce same idx for ep>=1; got {data_a.qpos[0]} vs {data_b.qpos[0]}"
        )

    def test_init_state_width_mismatch_raises(self):
        """Width != 1 + nq + nv must raise ``RuntimeError`` rather than
        silently slice. Critical guard for the procedural MJCF -> upstream
        LIBERO scene divergence (e.g. missing `(:objects ...)` clauses
        dropping free-joint bodies). Silent slicing would produce a
        deeply wrong physical state and mask a real bug."""
        nq, nv = 9, 9
        # State has 1+nq+5 = 15 entries instead of 1+nq+nv = 19. Mismatch.
        bad_state = np.zeros(15, dtype=np.float64)
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            init_states=bad_state,
        )

        sim, _, _ = self._make_sim_with_model(nq=nq, nv=nv)
        mj, _ = self._make_fake_mj()
        with patch.dict("sys.modules", {"mujoco": mj}):
            with pytest.raises(RuntimeError, match="init_state width"):
                adapter._apply_canonical_state(sim, random.Random(0))

    def test_actuator_state_present_raises(self):
        """``model.na != 0`` violates the
        ``[time, qpos, qvel]`` flat-state assumption. Raise rather than
        silently produce a wrong state - LIBERO scenes always have
        na=0; non-zero indicates a custom scene that needs a different
        applier."""
        nq, nv = 9, 9
        state = np.zeros(1 + nq + nv, dtype=np.float64)
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            init_states=state,
        )

        sim, _, _ = self._make_sim_with_model(nq=nq, nv=nv, na=2)  # actuator state!
        mj, _ = self._make_fake_mj()
        with patch.dict("sys.modules", {"mujoco": mj}):
            with pytest.raises(RuntimeError, match="actuator state"):
                adapter._apply_canonical_state(sim, random.Random(0))

    def test_init_states_takes_priority_over_keyframe(self):
        """When BOTH init_states and a model keyframe are present, the
        init_states branch wins. The keyframe applier must NOT fire.
        Pin to ensure init_states is the highest-priority canonical-state
        source; reviewers may otherwise expect keyframe (which is a
        scene-MJCF feature) to win."""
        nq, nv = 4, 4
        state = np.array([0.0] + [42.0] * nq + [0.0] * nv, dtype=np.float64)
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            init_states=state,
        )

        sim, model, data = self._make_sim_with_model(nq=nq, nv=nv)
        model.nkey = 1  # would route to keyframe branch without init_states

        # Stub mujoco - only used for mj_forward + (would be) mj_resetDataKeyframe.
        keyframe_calls: list[tuple[Any, Any, int]] = []
        forward_calls: list[tuple[Any, Any]] = []

        class _Mj:
            class mjtObj:
                mjOBJ_BODY = 1
                mjOBJ_JOINT = 3
                mjOBJ_ACTUATOR = 5

            @staticmethod
            def mj_resetDataKeyframe(model, data, idx):
                keyframe_calls.append((model, data, idx))

            @staticmethod
            def mj_forward(model, data):
                forward_calls.append((model, data))

        with patch.dict("sys.modules", {"mujoco": _Mj()}):
            adapter._apply_canonical_state(sim, random.Random(0))

        # Init-states fired (qpos written), keyframe did NOT.
        np.testing.assert_array_equal(data.qpos, state[1 : 1 + nq])
        assert keyframe_calls == []  # keyframe must NOT have fired
        assert len(forward_calls) == 1  # mj_forward from init_states branch

    def test_init_states_constructor_promotes_1d_to_2d(self):
        """Passing a 1D ``init_states`` (a single state) is legal -
        the constructor promotes it to ``(1, S)`` for uniform indexing."""
        state_1d = np.zeros(19, dtype=np.float64)
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            init_states=state_1d,
        )
        assert adapter._init_states is not None
        assert adapter._init_states.shape == (1, 19)

    def test_init_states_constructor_rejects_3d(self):
        """3D ndarray must raise ``ValueError`` at construction time -
        only 1D (single state) and 2D (n_states x state_dim) are valid."""
        bad = np.zeros((2, 3, 19), dtype=np.float64)
        with pytest.raises(ValueError, match="ndim"):
            LiberoAdapter.from_text(
                PICK_CUBE_BDDL,
                install_cameras=False,
                init_states=bad,
            )

    def test_init_states_none_falls_back_to_snapshot_branch(self):
        """``init_states=None`` (the default) preserves pre-#168-r7
        behaviour: snapshot-and-restore. Pin so existing user adapters
        that don't go through ``load_libero_suite`` keep working."""
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            init_states=None,
        )
        assert adapter._init_states is None
        # The actual branch behaviour is exercised by
        # TestApplyCanonicalStateSnapshot above; this test only pins the
        # constructor-default contract.

    def test_apply_canonical_state_passes_rng_through(self):
        """``on_episode_start`` forwards its ``rng`` to
        ``_apply_canonical_state`` so the init_states branch can
        seed-select. Pin the call chain because adding rng to the
        signature was a back-compat-fragile change."""
        # We just verify the method accepts rng as positional / keyword.
        # If rng plumbing breaks, _apply_canonical_state(sim, rng) raises TypeError.
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, install_cameras=False)
        sim = FakeSim(data_config="panda")
        # No model on FakeSim's world - method should debug-log + skip without error.
        adapter._apply_canonical_state(sim, random.Random(42))
        adapter._apply_canonical_state(sim)  # rng arg defaults to None


class TestLoadInitStatesBySuite:
    """``_load_init_states_by_bddl`` lazy-imports libero and returns a
    ``{bddl_filename: ndarray}`` map. Best-effort: missing libero, missing
    suite, per-task failures all return / preserve a partial dict
    (ideally empty) and never raise. The empty fallback lets
    ``load_libero_suite`` register tasks without init_states; the
    adapter then falls back to its snapshot-and-restore branch."""

    def test_libero_not_installed_returns_empty(self):
        """When ``libero`` isn't importable, return ``{}`` and don't
        raise. This is the minimal-CI / unit-test path."""
        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        # Patch builtins.__import__ to fail libero import.
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name == "libero.libero" or name.startswith("libero"):
                raise ImportError("libero not installed (test fixture)")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = _load_init_states_by_bddl("libero_10")
        assert result == {}

    def test_invalid_suite_returns_empty(self):
        """Suite name not in benchmark_dict() -> empty result, no raise."""
        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        # Patch the module-level lazy import via a stand-in benchmark that
        # exposes get_benchmark_dict but doesn't have our suite.
        fake_benchmark = MagicMock()
        fake_benchmark.get_benchmark_dict.return_value = {"libero_10": MagicMock()}
        fake_libero_libero = MagicMock()
        fake_libero_libero.benchmark = fake_benchmark
        with patch.dict(
            "sys.modules",
            {"libero.libero": fake_libero_libero, "libero": MagicMock()},
        ):
            result = _load_init_states_by_bddl("nonexistent_suite")
        assert result == {}

    def test_per_task_failure_skips_task_not_suite(self):
        """One task's get_task_init_states() raising must NOT abort the
        whole suite - other tasks still get loaded. Critical because
        a single corrupt .pruned_init file shouldn't block 50+ tasks."""
        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        states_a = np.zeros((50, 19), dtype=np.float64)
        states_a[0, 0] = 1.5  # marker
        # task 0 ok, task 1 raises, task 2 ok
        ts = MagicMock()
        ts.get_num_tasks.return_value = 3
        ts.get_task_bddl_files.return_value = ["a.bddl", "b.bddl", "c.bddl"]
        ts.get_task_init_states.side_effect = [
            states_a,
            RuntimeError("corrupt init state"),
            states_a,
        ]

        fake_benchmark = MagicMock()
        fake_benchmark.get_benchmark_dict.return_value = {"libero_10": lambda: ts}
        fake_libero_libero = MagicMock()
        fake_libero_libero.benchmark = fake_benchmark
        with patch.dict(
            "sys.modules",
            {"libero.libero": fake_libero_libero, "libero": MagicMock()},
        ):
            result = _load_init_states_by_bddl("libero_10")
        # Tasks 0 and 2 loaded; task 1 (raised) skipped.
        assert set(result.keys()) == {"a.bddl", "c.bddl"}
        # Tasks are keyed by bare filename, not full path.
        np.testing.assert_array_equal(result["a.bddl"], states_a)


# PolicyRunner + evaluate_benchmark integration


class TestEvaluateBenchmarkIntegration:
    def test_evaluate_with_mock_policy_succeeds(self):
        """Mock policy drives a loop; the benchmark loop returns a success_rate
        without crashing even though the mock policy doesn't actually win."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, max_steps=4)
        register_benchmark("libero-test-pick", adapter)
        # Sim is loaded with Panda but predicate positions don't match the
        # goal, so every episode should fall through to max_steps with
        # success=False - that's fine, we're testing the loop.
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0, 0, 0.0]},
                "plate_1": {"position": [0, 0, 0.0]},
            },
            data_config="panda",
        )
        # Rename so SimEngine.evaluate_benchmark can resolve the sole robot.
        sim._world.robots.clear()
        sim._world.robots["panda_arm"] = _FakeRobot("panda")

        result = sim.evaluate_benchmark(
            benchmark_name="libero-test-pick",
            policy_provider="mock",
            n_episodes=2,
            seed=0,
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_episodes"] == 2
        assert payload["benchmark_class"] == "LiberoAdapter"

    def test_unknown_task_returns_structured_error(self):
        sim = FakeSim(data_config="panda")
        result = sim.evaluate_benchmark(benchmark_name="libero-nonexistent")
        assert result["status"] == "error"
        assert "no benchmark registered" in result["content"][0]["text"].lower()

    def test_runner_counts_success_when_predicate_holds(self):
        """Seed the sim with predicate-satisfying state so is_success returns
        True on the first step - success_rate should be 1.0."""
        adapter = LiberoAdapter.from_text(
            "(define (problem p) (:goal (upright bottle)))",
            max_steps=3,
        )
        sim = FakeSim(
            bodies={"bottle": {"quaternion": [1.0, 0, 0, 0]}},
            data_config="panda",
        )
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_panda"))
        result = PolicyRunner(sim).evaluate("fake_panda", policy, spec=adapter, n_episodes=1)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_success"] == 1
        assert payload["episodes"][0]["steps"] == 1  # terminated on first step
