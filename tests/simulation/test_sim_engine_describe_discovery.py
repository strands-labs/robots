"""SimEngine.describe() exposes a single-call discovery surface.

Verifies that describe() lets a caller learn an engine's contract in one call
(its robots, cameras, world state, and the core method set) instead of
probe-and-fail. Covers the abstract base class via a minimal concrete engine
and the live MuJoCo engine.

Also pins the no-alias rule: the registry must NOT export a duplicate
``get_robot_info`` name. The canonical accessor is ``get_robot``; agents learn
it from the discovery surface rather than the API carrying a second name.
"""

from collections.abc import Sequence
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
            keyframe: str | int | None = None,
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
            material: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {}

        def remove_object(self, name: str) -> dict[str, Any]:
            return {}

        def get_observation(self, robot_name: str | None = None, **kw: Any) -> dict[str, Any]:
            return {}

        def send_action(
            self,
            action: dict[str, Any] | Sequence[float],
            robot_name: str | None = None,
            n_substeps: int = 1,
        ) -> dict[str, Any]:
            return {}

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

    def test_describe_lists_rollout_family_siblings(self):
        """describe() advertises the whole rollout family, not just run_policy.

        ``run_policy`` and ``start_policy`` were discoverable, but their
        siblings ``eval_policy`` (multi-episode success-rate evaluation) and
        ``replay_episode`` (replay a recorded LeRobotDataset episode) were not
        - so a caller enumerating ``describe()["methods"]`` could not learn the
        evaluation or replay entry points without guessing the names. They are
        concrete backend-agnostic facades on the base engine and belong on the
        discovery surface alongside ``run_policy``.
        """
        engine = _make_minimal_engine()
        methods = engine.describe()["methods"]
        for name in ("run_policy", "start_policy", "eval_policy", "replay_episode"):
            assert name in methods, f"describe() omits rollout-family method {name!r}"
        # Advertised signatures name the real first parameters so a caller can
        # invoke them without reading the source.
        assert "n_episodes" in methods["eval_policy"]
        assert "repo_id" in methods["replay_episode"]

    def test_describe_lists_scene_construction_methods(self):
        """describe() advertises how to build a scene, not just run one.

        Every rollout begins by constructing a scene -- add_robot, then
        add_object for manipulanda. These are concrete/abstract methods on the
        base contract, but describe() previously listed only the runtime
        (observe/act/rollout) surface, so a caller enumerating
        ``describe()["methods"]`` could not learn how to populate the world
        without guessing the names. They belong on the discovery surface as the
        first step a caller takes.
        """
        engine = _make_minimal_engine()
        methods = engine.describe()["methods"]
        for name in ("add_robot", "add_object", "remove_object"):
            assert name in methods, f"describe() omits scene-construction method {name!r}"
        # Advertised signatures name real parameters so a caller can invoke
        # them without reading the source.
        assert "urdf_path" in methods["add_robot"]
        assert "shape" in methods["add_object"]


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
            for name in ("eval_policy", "replay_episode"):
                assert name in desc["methods"], f"describe() omits {name!r}"
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

    def test_describe_lists_render_siblings(self):
        """describe() must advertise the full render surface, not just render().

        ``render_depth`` and ``render_all`` are public MuJoCo methods that the
        tool spec and action dispatcher already expose, but the programmatic
        discovery surface previously listed only ``render`` - so a caller
        enumerating ``describe()["methods"]`` could not learn that depth and
        multi-view rendering exist without guessing the names. They belong
        alongside ``render`` so one describe() call reveals the whole surface.
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in ("render", "render_depth", "render_all"):
                assert name in methods, f"describe() omits public render method {name!r}"
            # The advertised signatures name the real first parameters so a
            # caller can invoke them without reading the source.
            assert "camera_name" in methods["render_depth"]
            assert "cameras" in methods["render_all"]
        finally:
            sim.destroy()

    def test_describe_lists_scene_construction_methods(self):
        """describe() advertises the full scene-construction surface on MuJoCo.

        The documented rollout workflow is create_world -> add_robot ->
        add_object -> add_camera -> run_policy. add_camera/remove_camera are
        MuJoCo methods the tool spec already exposes, but the programmatic
        discovery surface omitted them (and add_object/remove_object), so a
        caller could not learn how to build the camera rig or place a cube from
        one describe() call.
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in ("add_robot", "add_object", "remove_object", "add_camera", "remove_camera"):
                assert name in methods, f"describe() omits scene-construction method {name!r}"
            # Advertised signatures name the real distinguishing parameters.
            assert "parent_body" in methods["add_camera"]
            assert "shape" in methods["add_object"]
        finally:
            sim.destroy()

    def test_describe_lists_physics_introspection_methods(self):
        """describe() advertises the read/verify surface, not just act/record.

        The discovery surface teaches how to build a scene, run a policy, and
        record a dataset, but previously listed no way to READ the physics
        result -- so an agent that ran a rollout could not learn how to verify
        it (read a body's world pose, check gripper-object contact, query a
        sensor) from one describe() call. These are public MuJoCo methods the
        tool spec + action dispatcher already expose; grounding a claim on a
        body-state delta (rather than a rendered caption) is the documented way
        to verify a rollout, so the primitives that produce that delta belong on
        the discovery surface alongside run_policy / start_recording.
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in (
                "get_body_state",
                "forward_kinematics",
                "get_contacts",
                "get_contact_forces",
                "get_sensor_data",
                "get_energy",
                "get_mass_matrix",
                "inverse_dynamics",
                "get_jacobian",
                "get_total_mass",
                "raycast",
                "multi_raycast",
            ):
                assert name in methods, f"describe() omits physics-introspection method {name!r}"
            # Advertised signatures name the real distinguishing parameters so a
            # caller can invoke them without reading the source.
            assert "body_name" in methods["get_body_state"]
            assert "sensor_name" in methods["get_sensor_data"]
            assert "origin" in methods["raycast"]
        finally:
            sim.destroy()

    def test_describe_start_recording_signature_includes_camera_scope(self):
        """describe()'s start_recording signature names the cameras= scope.

        start_recording grew a ``cameras=`` parameter that scopes the recorded
        LeRobotDataset to a subset of the scene's cameras, but the advertised
        signature omitted it -- so an agent enumerating describe() could not
        learn how to record a camera-scoped dataset without reading the source.
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            assert "cameras" in sim.describe()["methods"]["start_recording"]
        finally:
            sim.destroy()

    def test_describe_methods_resolve_to_real_attributes(self):
        """Every method MuJoCo describe() advertises must be a real callable.

        Pins the discovery-surface-is-a-contract invariant on the live engine:
        an advertised name that is not callable would dead-end a caller in an
        AttributeError instead of a working call.
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in methods:
                assert callable(getattr(sim, name, None)), (
                    f"describe() advertises {name!r} but it is not a callable on the engine"
                )
        finally:
            sim.destroy()


class TestNoAlias:
    """Code is the single source of truth: no duplicate-name aliases.

    The canonical registry accessor is ``get_robot``. Agents learn it from the
    discovery surface, so the API must NOT also export ``get_robot_info`` (a
    name an agent once guessed). One name per concept.
    """

    def test_no_get_robot_info_alias(self):
        import strands_robots.registry as registry

        assert not hasattr(registry, "get_robot_info"), (
            "registry must not export 'get_robot_info' - 'get_robot' is the "
            "single canonical name. Agents learn it via the discovery surface, "
            "not by aliasing a wrong guess."
        )
        assert "get_robot_info" not in getattr(registry, "__all__", [])
