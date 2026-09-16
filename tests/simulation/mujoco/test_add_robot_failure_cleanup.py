"""A failed ``add_robot`` must never leak a phantom robot into the world.

``Simulation.add_robot`` mutates two pieces of state before it can know the
load will succeed: it registers the ``SimRobot`` in ``world.robots[name]`` and
composes the robot into the live ``MjSpec`` (``inject_robot_into_scene``), which
recompiles the model. Any failure after registration - an asset that cannot be
downloaded, a scene injection the recompiler refuses, or an unexpected
exception from the injection layer - must roll the registry back to its
pre-call state and surface a structured error. Otherwise a half-added robot
lingers in ``world.robots`` and every downstream consumer (``list_robots``,
``get_observation``, recording) trips over a robot whose joints/actuators were
never wired into the compiled model, and re-adding the same name collides.

The observable proof of a clean rollback is that the *same name* adds
successfully right after a failed attempt: a leaked registry entry or an orphan
left in the spec would make the retry error out. These tests pin the three
cleanup branches; each fails if its ``pop``/``del`` were dropped.

A third piece of per-call state used to outlive a failed call: the notice that
the model was resolved through the deprecated name-as-registry-key fallback.
It was armed during resolution and disarmed only by the success return, so
every error exit in between left it armed on the engine and the NEXT successful
``add_robot`` printed it - telling a caller who passed ``data_config=``
correctly that they had used a deprecated form, and naming the earlier robot
rather than the one just added. The notice is call-local now; these tests pin
that it reaches the call that earned it and no other.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import simulation as sim_mod  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_add_robot_cleanup", mesh=False)
    s.create_world()
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


class TestAddRobotFailureCleanup:
    def test_mesh_asset_failure_returns_error_and_leaks_nothing(self, sim, monkeypatch):
        # An asset the auto-downloader cannot resolve is reported back as a
        # structured error dict from _ensure_meshes; add_robot must relay it
        # without registering the robot.
        err = {"status": "error", "content": [{"text": "asset download failed"}]}
        monkeypatch.setattr(sim, "_ensure_meshes", lambda *a, **k: err)

        result = sim.add_robot("so100")

        assert result["status"] == "error"
        assert result["content"][0]["text"] == "asset download failed"
        assert "so100" not in sim._world.robots

    def test_inject_refusal_rolls_back_registry_and_same_name_is_reusable(self, sim, monkeypatch):
        # inject_robot_into_scene returns False when the recompile is refused.
        # Force that once, then let the real injector run on retry: the retry
        # succeeds only if the failed attempt left no registry/spec residue.
        real_inject = sim_mod.inject_robot_into_scene
        calls = {"n": 0}

        def flaky_inject(world, robot, path, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return False
            return real_inject(world, robot, path, **kwargs)

        monkeypatch.setattr(sim_mod, "inject_robot_into_scene", flaky_inject)

        first = sim.add_robot("so100")
        assert first["status"] == "error"
        assert "so100" not in sim._world.robots

        retry = sim.add_robot("so100")
        assert retry["status"] == "success"
        assert "so100" in sim._world.robots

    def test_unexpected_exception_pops_registry_and_same_name_is_reusable(self, sim, monkeypatch):
        # An unexpected exception from the injection layer (e.g. a MuJoCo
        # compile crash, not a clean False) must be caught, reported, and the
        # partially-registered robot popped - not propagated to the caller.
        real_inject = sim_mod.inject_robot_into_scene
        calls = {"n": 0}

        def flaky_inject(world, robot, path, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated compile crash")
            return real_inject(world, robot, path, **kwargs)

        monkeypatch.setattr(sim_mod, "inject_robot_into_scene", flaky_inject)

        first = sim.add_robot("so100")
        assert first["status"] == "error"
        assert "Failed to load" in first["content"][0]["text"]
        assert "so100" not in sim._world.robots

        retry = sim.add_robot("so100")
        assert retry["status"] == "success"
        assert "so100" in sim._world.robots


class TestDeprecationNoticeIsPerCall:
    """The name-as-registry-key notice reaches only the call that earned it."""

    @staticmethod
    def _fail_ensure_meshes(sim, monkeypatch):
        monkeypatch.setattr(
            sim,
            "_ensure_meshes",
            lambda *a, **k: {"status": "error", "content": [{"text": "asset download failed"}]},
        )

    @staticmethod
    def _refuse_inject(sim, monkeypatch):
        monkeypatch.setattr(sim_mod, "inject_robot_into_scene", lambda *a, **k: False)

    @staticmethod
    def _crash_inject(sim, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("simulated compile crash")

        monkeypatch.setattr(sim_mod, "inject_robot_into_scene", boom)

    @pytest.mark.parametrize(
        "break_it",
        [_fail_ensure_meshes, _refuse_inject, _crash_inject],
        ids=["mesh_asset_failure", "inject_refusal", "inject_crash"],
    )
    def test_a_failed_deprecated_add_does_not_warn_the_next_caller(self, sim, monkeypatch, break_it):
        # A deprecated positional add that fails after resolution...
        break_it.__func__(sim, monkeypatch)
        failed = sim.add_robot("so100")
        assert failed["status"] == "error"

        # ...must not put its notice on the next call, which named an instance
        # label and passed the model through data_config= exactly as advised.
        monkeypatch.undo()
        clean = sim.add_robot(name="arm_two", data_config="so100")
        assert clean["status"] == "success"
        text = clean["content"][0]["text"]
        assert "Warning" not in text, f"stale notice leaked into a correct call: {text}"
        assert "deprecated" not in text
        # The leaked notice also named the wrong robot: the one that failed.
        assert "so100'" not in text.replace("data_config='so100'", "")

    def test_the_deprecated_form_that_succeeds_still_warns(self, sim):
        # Essence check: dropping the engine attribute must not drop the notice.
        result = sim.add_robot("so100")

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Warning" in text
        assert "deprecated name-as-registry-key fallback" in text
        assert "add_robot(name='<instance_label>', data_config='so100')" in text

    def test_a_correct_add_never_warns(self, sim):
        result = sim.add_robot(name="arm_one", data_config="so100")

        assert result["status"] == "success"
        assert "Warning" not in result["content"][0]["text"]
