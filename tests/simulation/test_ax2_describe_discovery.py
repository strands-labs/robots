"""AX-2 regression test: SimEngine.describe() discovery surface.

Verifies the discovery surface so agents can learn an engine's contract in a
single call instead of probe-and-fail. Pre-fix, describe() does not exist and
these assertions fail with AttributeError.

Also pins the no-alias rule: the registry must NOT export a duplicate
``get_robot_info`` name. The canonical accessor is ``get_robot``; agents learn
it from the discovery surface rather than the API carrying a second name.
"""

from typing import Any

import pytest


def _make_minimal_engine():
    """Create a minimal SimEngine with one robot for describe() testing."""
    from strands_robots.simulation.base import SimEngine

    class MinimalEngine(SimEngine):
        """Smallest concrete engine to test describe()."""

        def __init__(self) -> None:
            self._robots: list[str] = []

        def create_world(
            self,
            timestep: float | None = None,
            gravity: list[float] | None = None,
            ground_plane: bool = True,
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
            **kwargs: Any,
        ) -> dict[str, Any]:
            self._robots.append(name)
            return {}

        def remove_robot(self, name: str) -> dict[str, Any]:
            self._robots.remove(name)
            return {}

        def list_robots(self) -> list[str]:
            return list(self._robots)

        def robot_joint_names(self, robot_name: str) -> list[str]:
            return ["joint_0", "joint_1"]

        def add_object(
            self,
            name: str,
            shape: str = "box",
            position: list[float] | None = None,
            orientation: list[float] | None = None,
            size: list[float] | None = None,
            color: list[float] | None = None,
            mass: float = 1.0,
            is_static: bool = False,
            mesh_path: str | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {}

        def remove_object(self, name: str) -> dict[str, Any]:
            return {}

        def get_observation(self, robot_name: str | None = None, **kw: Any) -> dict[str, Any]:
            return {}

        def send_action(
            self,
            action: dict[str, Any],
            robot_name: str | None = None,
            n_substeps: int = 1,
        ) -> None:
            pass

        def render(
            self,
            camera_name: str = "default",
            width: int | None = None,
            height: int | None = None,
        ) -> dict[str, Any]:
            return {}

    return MinimalEngine()


class TestDescribeABC:
    """Tests for SimEngine.describe() on the abstract base class."""

    def test_describe_exists(self):
        engine = _make_minimal_engine()
        result = engine.describe()
        assert isinstance(result, dict)

    def test_describe_robots_equals_list_robots(self):
        engine = _make_minimal_engine()
        engine.add_robot("test_bot")
        desc = engine.describe()
        assert desc["robots"] == engine.list_robots()
        assert desc["robots"] == ["test_bot"]

    def test_describe_robots_empty_when_no_robots(self):
        engine = _make_minimal_engine()
        desc = engine.describe()
        assert desc["robots"] == []

    def test_describe_has_get_robot_state_key(self):
        engine = _make_minimal_engine()
        desc = engine.describe()
        assert "methods" in desc
        assert "get_robot_state" in desc["methods"]

    def test_describe_has_cameras_key(self):
        engine = _make_minimal_engine()
        desc = engine.describe()
        assert "cameras" in desc
        assert isinstance(desc["cameras"], list)

    def test_describe_has_note(self):
        engine = _make_minimal_engine()
        desc = engine.describe()
        assert "note" in desc
        assert "robot_name" in desc["note"]

    def test_describe_methods_includes_core_set(self):
        engine = _make_minimal_engine()
        desc = engine.describe()
        expected_methods = {
            "get_robot_state",
            "get_observation",
            "send_action",
            "run_policy",
            "list_robots",
            "render",
        }
        assert expected_methods.issubset(set(desc["methods"].keys()))


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="MuJoCo not installed"),
    reason="MuJoCo not available",
)
class TestDescribeMuJoCo:
    """Tests for MuJoCoSimEngine.describe() with a live sim world."""

    def test_describe_with_world_and_robot(self):
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            sim.create_world()
            sim.add_robot("so100", data_config="so100")
            desc = sim.describe()

            assert desc["robots"] == sim.list_robots()
            assert "so100" in desc["robots"]
            assert desc["world_created"] is True
            assert isinstance(desc["cameras"], list)
            assert "get_robot_state" in desc["methods"]
        finally:
            sim.destroy()

    def test_describe_no_world(self):
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        desc = sim.describe()
        assert desc["robots"] == []
        assert desc["world_created"] is False


class TestNoAlias:
    """Code is the single source of truth: no duplicate-name aliases.

    The canonical registry accessor is ``get_robot``. Agents learn it from the
    discovery surface, so the API must NOT also export ``get_robot_info`` (a
    name an agent once guessed). One name per concept.
    """

    def test_no_get_robot_info_alias(self):
        import strands_robots.registry as registry

        assert not hasattr(registry, "get_robot_info"), (
            "registry must not export 'get_robot_info' — 'get_robot' is the "
            "single canonical name. Agents learn it via the discovery surface, "
            "not by aliasing a wrong guess."
        )
        assert "get_robot_info" not in getattr(registry, "__all__", [])
