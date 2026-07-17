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

    def test_describe_lists_get_state(self):
        """describe() advertises get_state, the whole-world snapshot method.

        ``get_state`` is an abstract method every backend implements and a
        first-class action in the tool spec, but the discovery surface listed
        only the per-robot readers (``get_robot_state`` / ``get_observation``)
        - so a caller enumerating ``describe()["methods"]`` could not learn how
        to read the world-level snapshot (sim time, step count, entity counts)
        without guessing the name. It belongs on the base contract alongside
        the other read methods.
        """
        engine = _make_minimal_engine()
        methods = engine.describe()["methods"]
        assert "get_state" in methods, "describe() omits the base get_state method"

    def test_describe_lists_benchmark_family_methods(self):
        """describe() advertises the DSL-driven benchmark scoring surface.

        ``run_policy``/``eval_policy`` were discoverable, but their DSL-scored
        siblings were not: ``evaluate_benchmark`` (score a registered
        success/failure/dense_reward benchmark over a rollout),
        ``list_benchmarks`` (enumerate the registered benchmark names it
        accepts), and ``register_benchmark_from_file`` (author a benchmark spec
        as YAML/JSON at runtime). These are concrete backend-agnostic facades on
        the base engine, so a caller enumerating ``describe()["methods"]`` could
        run a policy but could not discover how to score it against a benchmark
        without guessing the names.
        """
        engine = _make_minimal_engine()
        methods = engine.describe()["methods"]
        for name in ("evaluate_benchmark", "list_benchmarks", "register_benchmark_from_file"):
            assert name in methods, f"describe() omits benchmark-family method {name!r}"
        # Advertised signatures name the real distinguishing parameters so a
        # caller can invoke them without reading the source.
        assert "benchmark_name" in methods["evaluate_benchmark"]
        assert "spec_path" in methods["register_benchmark_from_file"]

    def test_describe_register_builtin_benchmarks_advertises_humanoids(self):
        """describe() advertises register_builtin_benchmarks as shipping BOTH the
        quadruped and the humanoid locomotion tasks, not the quadruped alone.

        register_builtin_benchmarks() ships three velocity-tracking tasks -
        go2_walk_forward (quadruped) plus g1_walk_forward and t1_walk_forward
        (two humanoids of different scale). The discovery-surface description
        once named only go2_walk_forward as its example, so an agent enumerating
        describe()["methods"] to decide whether a runnable HUMANOID locomotion
        eval exists would be misled into thinking only the quadruped ships. Pin
        the humanoid tasks into the advertised text so it cannot silently
        regress to quadruped-only as more built-ins are added.
        """
        engine = _make_minimal_engine()
        blurb = engine.describe()["methods"]["register_builtin_benchmarks"]
        assert "g1_walk_forward" in blurb
        assert "t1_walk_forward" in blurb
        assert "humanoid" in blurb.lower()

    def test_describe_lists_scene_mutation_inverse(self):
        """describe() advertises remove_robot, completing the add/remove pair.

        The base contract advertised ``add_robot`` ("the first
        scene-construction step") and the object inverse ``remove_object``, but
        not ``remove_robot`` -- even though it is an abstract method every
        backend must implement. A caller enumerating ``describe()["methods"]``
        who built a scene with add_robot could not learn how to tear a robot
        back out without guessing the name. It belongs on the base discovery
        surface as the inverse of add_robot, mirroring the add_object /
        remove_object pair already advertised.
        """
        engine = _make_minimal_engine()
        methods = engine.describe()["methods"]
        assert "remove_robot" in methods, "describe() omits the base remove_robot method"
        assert "name" in methods["remove_robot"]

    def test_describe_lists_scene_variation_and_grounding(self):
        """describe() advertises the scene-variation + physics-grounding facades.

        ``load_scene`` (the alternative scene-construction entry point),
        ``randomize`` and ``set_obs_noise`` (domain randomization + sensor-noise
        variation), and ``get_contacts`` (the physics-grounding read used to
        verify a grasp / detect a collision) are all public methods declared on
        the base ``SimEngine`` contract, but describe() listed none of them -- so
        a caller enumerating ``describe()["methods"]`` could build a scene and
        run a policy, yet could not discover how to load a scene from file, vary
        it for sim2real, or verify the physics result without guessing the
        names. They belong on the base discovery surface; each backend documents
        its concrete signature via its own describe() override.
        """
        engine = _make_minimal_engine()
        methods = engine.describe()["methods"]
        for name in ("load_scene", "randomize", "set_obs_noise", "get_contacts"):
            assert name in methods, f"describe() omits base scene/grounding method {name!r}"
        # Advertised signatures name the real distinguishing details so a caller
        # can invoke them (or know where to read the concrete backend signature).
        assert "scene_path" in methods["load_scene"]
        assert "contact" in methods["get_contacts"].lower()

    def test_describe_lists_world_lifecycle_entry_points(self):
        """describe() advertises create_world and destroy on the BASE contract.

        create_world (the fresh-world entry point that precedes add_robot /
        add_object) and destroy (its teardown inverse) are abstract methods
        every backend must implement, and their lifecycle siblings reset / step
        / get_state were already advertised. But the base discovery surface
        listed neither create_world nor destroy -- so a caller enumerating
        ``describe()["methods"]`` on any backend that does not separately re-add
        them (the Newton engine builds its surface from scratch and omitted them
        too) could not learn how to CREATE the world it must populate, nor how
        to tear it down, without guessing. They belong on the base contract
        beside reset / step / get_state as the world-lifecycle entry points.
        """
        engine = _make_minimal_engine()
        methods = engine.describe()["methods"]
        for name in ("create_world", "destroy"):
            assert name in methods, f"describe() omits world-lifecycle method {name!r}"
        # The advertised create_world signature names the real knobs (the flat
        # floor toggle and the locomotion-terrain curriculum) so a caller can
        # spawn a robot on non-flat ground without reading the source.
        blurb = methods["create_world"]
        assert "ground_plane" in blurb
        assert "terrain" in blurb
        assert "difficulty" in blurb


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

    def test_describe_lists_scene_and_object_manipulation_siblings(self):
        """describe() advertises load_scene and the object-manipulation siblings.

        describe() advertised add_object / remove_object and add_robot, but
        omitted the alternative scene-construction entry point (load_scene), the
        object siblings that complete the add/remove pair (list_objects,
        move_object), and the domain-randomization facade (randomize) -- all
        first-class public methods the tool spec + action dispatcher already
        expose. An agent enumerating how to build and vary a scene from
        describe() alone could not discover them and had to guess method names.
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in ("load_scene", "list_objects", "move_object", "randomize"):
                assert name in methods, f"describe() omits scene/object method {name!r}"
            # Advertised signatures name the real distinguishing parameters so a
            # caller can invoke them without reading the source.
            assert "scene_path" in methods["load_scene"]
            assert "orientation" in methods["move_object"]
            assert "randomize_physics" in methods["randomize"]
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

    def test_describe_lists_sim_state_family(self):
        """describe() advertises the sim-state checkpoint + pose-setting family.

        ``save_state`` / ``load_state`` (checkpoint and restore the whole
        physics state) and ``set_joint_positions`` / ``set_joint_velocities``
        (write qpos/qvel directly to teleport the robot to a pose or set an
        initial dynamic state) are first-class actions in the MuJoCo tool spec
        and action dispatcher, plus ``get_state`` from the base contract. But
        the discovery surface listed none of them -- so an agent setting up a
        deterministic initial condition, or A/B-testing two rollouts from the
        same checkpoint, had to guess these names. They belong on the discovery
        surface alongside the act / read surfaces.
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in (
                "save_state",
                "load_state",
                "set_joint_positions",
                "set_joint_velocities",
                "get_state",
            ):
                assert name in methods, f"describe() omits sim-state method {name!r}"
            # Advertised signatures name the real distinguishing parameters so a
            # caller can invoke them without reading the source.
            assert "name" in methods["save_state"]
            assert "positions" in methods["set_joint_positions"]
            assert "velocities" in methods["set_joint_velocities"]
        finally:
            sim.destroy()

    def test_describe_lists_physics_tuning_methods(self):
        """describe() advertises the physics-tuning / domain-perturbation WRITE surface.

        The complement of the physics-introspection READ family: ``set_gravity``
        / ``set_timestep`` (retune the engine), ``set_body_properties`` /
        ``set_geom_properties`` (per-body / per-geom domain randomization), and
        ``apply_force`` (an external wrench for push-recovery / perturbation
        testing) are all first-class actions in the MuJoCo tool spec and action
        dispatcher, and the engine's own guidance points a caller at
        "set_gravity, set_timestep, etc." -- but the discovery surface listed
        none of them, so an agent setting up a domain-randomization / sim2real
        scene had to guess these names. They belong on the discovery surface
        alongside the randomize() facade and the physics-read surface.
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in (
                "set_gravity",
                "set_timestep",
                "set_body_properties",
                "set_geom_properties",
                "apply_force",
            ):
                assert name in methods, f"describe() omits physics-tuning method {name!r}"
            # Advertised signatures name the real distinguishing parameters so a
            # caller can invoke them without reading the source.
            assert "gravity" in methods["set_gravity"]
            assert "timestep" in methods["set_timestep"]
            assert "body_name" in methods["set_body_properties"]
            assert "friction" in methods["set_geom_properties"]
            assert "torque" in methods["apply_force"]
        finally:
            sim.destroy()

    def test_describe_lists_cameras_recording_family(self):
        """describe() advertises the plain-MP4 recorder, not only the dataset one.

        The recording surface already advertises the LeRobotDataset family
        (``start_recording`` / ``save_episode`` / ``stop_recording``), which
        needs the ``[lerobot]`` extra and writes a parquet+MP4 dataset. But the
        dependency-free ``start_cameras_recording`` / ``stop_cameras_recording``
        / ``get_cameras_recording_status`` trio -- which writes one raw MP4 per
        camera with no lerobot dependency -- was undiscoverable from describe()
        alone, even though all three are first-class MuJoCo tool-spec and
        action-dispatcher actions. An agent lacking lerobot, or wanting a raw
        MP4, had to guess these names. They belong on the discovery surface as
        the raw-MP4 sibling of the dataset trio.
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in (
                "start_cameras_recording",
                "stop_cameras_recording",
                "get_cameras_recording_status",
            ):
                assert name in methods, f"describe() omits cameras-recording method {name!r}"
            # Advertised signatures name the real distinguishing parameters so a
            # caller can invoke them without reading the source.
            assert "cameras" in methods["start_cameras_recording"]
            assert "max_frames_per_camera" in methods["start_cameras_recording"]
        finally:
            sim.destroy()

    def test_describe_lists_robot_registry_family(self):
        """describe() advertises the robot-registry + remove_robot surface.

        describe() calls ``add_robot`` "the first scene-construction step" and
        advertises the object/camera remove halves (``remove_object`` /
        ``remove_camera``), but the robot inverse ``remove_robot`` -- and the
        registry methods that feed ``add_robot(name=...)`` (``list_urdfs`` to
        discover the registered names, ``register_urdf`` to add one) -- were
        undiscoverable from describe() alone, even though all three are
        first-class MuJoCo ``tool_spec.json`` + action-dispatcher actions. A
        caller who built a scene with add_robot could not learn how to remove a
        robot, or how to register a custom URDF, without guessing these names.
        They belong on the discovery surface, completing the add/remove symmetry
        that remove_object / remove_camera already establish.
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in ("list_urdfs", "register_urdf", "remove_robot"):
                assert name in methods, f"describe() omits robot-registry method {name!r}"
            # Advertised signatures name the real distinguishing parameters so a
            # caller can invoke them without reading the source.
            assert "data_config" in methods["register_urdf"]
            assert "urdf_path" in methods["register_urdf"]
            assert "name" in methods["remove_robot"]
        finally:
            sim.destroy()

    def test_describe_lists_scene_lifecycle_methods(self):
        """describe() advertises the world-lifecycle + MJCF-editing surface.

        describe() taught how to build a scene, run a policy, and read the
        result, but previously omitted the world lifecycle itself (create_world,
        the fresh-world entry point that precedes add_robot; destroy, which the
        tool-spec guidance explicitly asks callers to run at session end) and the
        MJCF-editing family (patch_scene_mjcf / replace_scene_mjcf, export_xml).
        All five are first-class actions in the tool spec + action dispatcher, so
        a caller enumerating how to create, edit, and tear down a scene from
        describe() alone had to guess these names. (The URDF/model registry trio
        -- register_urdf / list_urdfs / remove_robot -- is covered by
        test_describe_lists_robot_registry_family.)
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in (
                "create_world",
                "destroy",
                "patch_scene_mjcf",
                "replace_scene_mjcf",
                "export_xml",
            ):
                assert name in methods, f"describe() omits scene-lifecycle method {name!r}"
            # Advertised signatures name the real distinguishing parameters so a
            # caller can invoke them without reading the source.
            assert "ground_plane" in methods["create_world"]
            assert "ops" in methods["patch_scene_mjcf"]
            assert "xml" in methods["replace_scene_mjcf"]
            assert "output_path" in methods["export_xml"]
        finally:
            sim.destroy()

    def test_describe_lists_teleop_family(self):
        """describe() advertises the teleoperation surface, not only run_policy.

        describe() teaches how to build a scene and drive it with a policy
        (``run_policy`` / ``start_policy``), but the OTHER actuation source --
        driving a sim robot from an attached teleoperator (a real leader arm,
        gamepad, or keyboard), the leader->follower / human-demonstration
        workflow that feeds data collection -- was undiscoverable from
        describe() alone. The six ``TeleopMixin`` facades (``attach_teleop`` ->
        ``teleoperate`` -> ``stop_teleoperate``, plus ``detach_teleop`` /
        ``list_teleops`` / ``get_teleoperate_status``) are public methods on the
        sim, yet a caller had to guess their names. They belong on the discovery
        surface as the human-driven sibling of the policy-rollout family.
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in (
                "attach_teleop",
                "teleoperate",
                "stop_teleoperate",
                "get_teleoperate_status",
                "list_teleops",
                "detach_teleop",
            ):
                assert name in methods, f"describe() omits teleop method {name!r}"
            # Advertised signatures name the real distinguishing parameters so a
            # caller can invoke them without reading the source.
            assert "map_fn" in methods["attach_teleop"]
            assert "publish" in methods["teleoperate"]
            assert "duration" in methods["teleoperate"]
            assert "name" in methods["detach_teleop"]
        finally:
            sim.destroy()

    def test_describe_lists_viewer_family(self):
        """describe() advertises the interactive-viewer surface.

        describe() taught how to build a scene, drive it with a policy, and read
        the result, but gave no way to discover how to OPEN a live window on the
        running model for human inspection (watch a rollout, debug a pose,
        hand-verify a scene). open_viewer / close_viewer are first-class actions
        in the tool spec + action dispatcher, so a caller enumerating the sim's
        contract from describe() alone had to guess their names. The advertised
        open_viewer signature also documents the headless caveat (needs a local
        display; render()/render_all() capture frames instead).
        """
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            methods = sim.describe()["methods"]
            for name in ("open_viewer", "close_viewer"):
                assert name in methods, f"describe() omits viewer method {name!r}"
            # The advertised open_viewer signature warns of the display
            # requirement so a caller does not blindly invoke it on a headless
            # host (where render()/render_all() are the right frame source).
            assert "display" in methods["open_viewer"]
            assert "render" in methods["open_viewer"]
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
