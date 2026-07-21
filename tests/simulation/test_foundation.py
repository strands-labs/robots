"""Tests for simulation foundation - models, ABC, factory, model_registry.

These tests verify the lightweight simulation abstractions without
requiring MuJoCo or any heavy dependencies.
"""

from collections.abc import Sequence
from typing import Any

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.factory import (
    create_simulation,
    list_backends,
    register_backend,
)
from strands_robots.simulation.models import (
    SimObject,
    SimRobot,
    SimStatus,
    SimWorld,
    TrajectoryStep,
)

# Shared fixtures


def _make_dummy_engine_class() -> type[SimEngine]:
    """Create a minimal concrete SimEngine subclass.

    All 12 required abstract methods return empty dicts / None.
    Factored out to avoid ~150 lines of repetition across tests.
    """

    class Dummy(SimEngine):
        def create_world(
            self,
            timestep: float | None = None,
            gravity: list[float] | None = None,
            ground_plane: bool = True,
            terrain: str | None = None,
            difficulty: float = 1.0,
        ) -> dict[str, Any]:
            return {}

        def destroy(self) -> dict[str, Any]:
            return {}

        def reset(self) -> dict[str, Any]:
            return {}

        def step(self, n_steps: int = 1) -> dict[str, Any]:
            return {}

        def get_state(self) -> dict[str, Any]:
            return {}

        def add_robot(
            self,
            name: str,
            urdf_path: str | None = None,
            data_config: str | None = None,
            position: list[float] | None = None,
            orientation: list[float] | None = None,
            keyframe: str | int | None = None,
        ) -> dict[str, Any]:
            return {}

        def remove_robot(self, name: str) -> dict[str, Any]:
            return {}

        def list_robots(self) -> list[str]:
            return []

        def robot_joint_names(self, robot_name: str) -> list[str]:
            return []

        def add_object(
            self,
            name: str,
            shape: str = "box",
            position: list[float] | None = None,
            orientation: list[float] | None = None,
            size: list[float] | None = None,
            color: list[float] | None = None,
            mass: float = 0.1,
            is_static: bool = False,
            mesh_path: str | None = None,
            material: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {}

        def remove_object(self, name: str) -> dict[str, Any]:
            return {}

        def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
            return {}

        def send_action(
            self, action: dict[str, Any] | Sequence[float], robot_name: str | None = None, n_substeps: int = 1
        ) -> dict[str, Any]:
            return {"status": "success", "content": [{"text": "ok"}]}

        def render(
            self, camera_name: str = "default", width: int | None = None, height: int | None = None
        ) -> dict[str, Any]:
            return {}

    return Dummy


@pytest.fixture
def dummy_engine_class() -> type[SimEngine]:
    """Fixture providing a minimal concrete SimEngine subclass."""
    return _make_dummy_engine_class()


# ABC Tests


class TestSimEngine:
    """Test the abstract base class contract."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SimEngine()

    def test_has_required_abstract_methods(self):
        abstract_methods = SimEngine.__abstractmethods__
        expected = {
            "create_world",
            "destroy",
            "reset",
            "step",
            "get_state",
            "add_robot",
            "remove_robot",
            "list_robots",
            "robot_joint_names",
            "add_object",
            "remove_object",
            "get_observation",
            "send_action",
            "render",
        }
        assert expected == abstract_methods

    def test_optional_methods_raise_not_implemented(self, dummy_engine_class):
        """Optional methods on a concrete subclass raise NotImplementedError.

        Note: ``run_policy`` / ``replay_episode`` / ``eval_policy`` used to
        be in this set but are now concrete facades on the ABC that
        delegate to the backend-agnostic ``PolicyRunner``.
        """
        d = dummy_engine_class()
        with pytest.raises(NotImplementedError):
            d.load_scene("x")
        with pytest.raises(NotImplementedError):
            d.randomize()
        with pytest.raises(NotImplementedError):
            d.get_contacts()

    def test_context_manager_calls_cleanup(self, dummy_engine_class):
        """ABC supports context manager protocol and calls cleanup on exit."""
        cleaned = {"flag": False}

        class Cleanable(dummy_engine_class):  # type: ignore[misc,valid-type]
            def cleanup(self) -> None:
                cleaned["flag"] = True

        with Cleanable():
            pass
        assert cleaned["flag"] is True


# Factory Tests


class TestSimulationFactory:
    """Test backend registration and creation - full round-trip."""

    def test_list_backends_includes_mujoco(self):
        backends = list_backends()
        assert "mujoco" in backends

    def test_register_create_and_use_backend(self, dummy_engine_class):
        """Register a custom backend, create it via factory, verify instance."""
        register_backend("fake_test", lambda: dummy_engine_class, force=True)
        assert "fake_test" in list_backends()
        sim = create_simulation("fake_test")
        assert isinstance(sim, dummy_engine_class)

    def test_register_rejects_duplicate(self, dummy_engine_class):
        """Registering an existing name without force raises ValueError."""
        register_backend("dup_test", lambda: dummy_engine_class, force=True)
        with pytest.raises(ValueError, match="already registered"):
            register_backend("dup_test", lambda: dummy_engine_class)

    def test_register_rejects_builtin_alias_in_aliases(self, dummy_engine_class):
        """Cannot use a built-in alias as a new backend's alias."""
        with pytest.raises(ValueError, match="conflicts with built-in"):
            register_backend("custom_phys", lambda: dummy_engine_class, aliases=["mj"])

    # Regression tests for alias-shadowing bug (PR #84 review)

    def test_register_rejects_builtin_alias_as_name(self, dummy_engine_class):
        """Cannot register a new backend under a built-in alias name.

        Regression test for the bug where ``register_backend("mj", loader)``
        succeeded without ``force=True`` because the conflict check only
        looked at ``_BUILTIN_BACKENDS`` and ``_runtime_registry``, missing
        ``_BUILTIN_ALIASES``.
        """
        for builtin_alias in ("mj", "mjc", "mjx"):
            with pytest.raises(ValueError, match="conflicts with built-in alias"):
                register_backend(builtin_alias, lambda: dummy_engine_class)

    def test_register_rejects_runtime_alias_as_name(self, dummy_engine_class):
        """Cannot register a new backend under a runtime-registered alias name."""
        register_backend("backend_a", lambda: dummy_engine_class, aliases=["short_a"], force=True)
        with pytest.raises(ValueError, match="conflicts with runtime alias"):
            register_backend("short_a", lambda: dummy_engine_class)

    def test_register_rejects_backend_name_as_alias(self, dummy_engine_class):
        """Cannot use an existing backend name as a new backend's alias."""
        with pytest.raises(ValueError, match="conflicts with existing backend name"):
            register_backend("new_x", lambda: dummy_engine_class, aliases=["mujoco"])

    def test_register_force_overrides_alias_conflict(self, dummy_engine_class):
        """force=True bypasses all conflict checks (escape hatch)."""
        # Should NOT raise
        register_backend("mj", lambda: dummy_engine_class, force=True)
        # Clean up - put the real mj alias back by re-importing
        import importlib

        from strands_robots.simulation import factory

        importlib.reload(factory)


# Model Registry Tests


class TestModelRegistry:
    """Test URDF/MJCF model resolution."""

    def test_list_available_models_returns_robot_table(self):
        from strands_robots.simulation.model_registry import list_available_models

        models = list_available_models()
        assert isinstance(models, str)
        assert "so100" in models
        assert len(models) > 100

    def test_register_and_resolve_urdf(self, tmp_path):
        """Register a URDF, resolve it back - full round-trip."""
        from strands_robots.simulation.model_registry import register_urdf, resolve_urdf

        urdf_file = tmp_path / "robot.urdf"
        urdf_file.write_text("<robot/>")
        register_urdf("test_robot_xyz", str(urdf_file))
        result = resolve_urdf("test_robot_xyz")
        assert result == str(urdf_file)

    def test_list_registered_urdfs(self):
        from strands_robots.simulation.model_registry import list_registered_urdfs, register_urdf

        register_urdf("list_test_bot", "/fake/list.urdf")
        urdfs = list_registered_urdfs()
        assert isinstance(urdfs, dict)
        assert "list_test_bot" in urdfs


# Dataclass Behavioral Tests


class TestSimModelsUsage:
    """Test that simulation models behave correctly in real usage patterns."""

    def test_sim_world_tracks_robots(self):
        """SimWorld can add robots and objects - simulates real world setup."""
        world = SimWorld()
        robot = SimRobot(name="so100", urdf_path="/p")
        world.robots["so100"] = robot
        assert "so100" in world.robots
        assert world.status == SimStatus.IDLE

    def test_sim_object_preserves_originals_for_randomization(self):
        """SimObject stores original position/color for domain randomization reset."""
        obj = SimObject(name="ball", shape="sphere", position=[1, 2, 3], color=[1, 0, 0, 1])
        assert obj._original_position == [1, 2, 3]
        assert obj._original_color == [1, 0, 0, 1]

    def test_trajectory_step_records_episode_data(self):
        """TrajectoryStep captures full observation-action pair for dataset recording."""
        step = TrajectoryStep(
            timestamp=1.0,
            sim_time=0.5,
            robot_name="arm",
            observation={"state": [1, 2, 3]},
            action={"joint_0": 0.5},
            instruction="pick up cube",
        )
        assert step.robot_name == "arm"
        assert step.instruction == "pick up cube"
        assert step.observation["state"] == [1, 2, 3]
