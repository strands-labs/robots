"""GPU integration tests for the Isaac Sim simulation backend.

Ported from ``strands-robots-sim`` (``strands_robots_sim/isaac/tests/
test_gpu_integ.py``) as part of the Isaac backend migration
(`#1148 <https://github.com/strands-labs/robots/issues/1148>`_, parent
`#1144 <https://github.com/strands-labs/robots/issues/1144>`_).

The Isaac Sim backend ships **out-of-tree** in the sibling
`strands-robots-sim <https://github.com/strands-labs/robots-sim>`_ plugin
package and registers an ``IsaacSimulation`` under the
``strands_robots.backends`` entry-point group (see
``docs/simulation/isaac.md``). These tests therefore exercise the plugin
through both the documented factory path
(``create_simulation("isaac", ...)``) and the direct constructor
(``IsaacSimulation(IsaacConfig(...))``), so the backend contract is pinned
regardless of which entry point a caller uses.

Requirements:
  - NVIDIA GPU with CUDA
  - Isaac Sim 6.0+ installed (Python 3.12), installed out-of-band
  - ``pip install 'strands-robots-sim[isaac]'`` (provides the plugin)
  - Environment variable: ``STRANDS_GPU_TEST=1``

Gated behind the ``gpu`` marker AND ``STRANDS_GPU_TEST=1`` so a CI run that
happens to have the plugin installed without a GPU never boots ``SimulationApp``
and times out. ``importorskip`` emits a clean SKIPPED line (not an opaque
collection-time ImportError) when the plugin is absent.

Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ \
        tests_integ/simulation/test_isaac_gpu.py -m gpu -v

These assert only user-visible behaviour (status envelopes, observation
shapes, RGB frame dtype/channels) - never internal simulation state - per the
"test behavior, not implementation" rule.
"""

from __future__ import annotations

import os

import pytest

# Skip cleanly if the Isaac plugin isn't installed. Isaac ships in
# strands-robots-sim, not this package, so this is the correct import surface
# for the migrated backend.
pytest.importorskip(
    "strands_robots_sim.isaac",
    reason="Isaac backend not installed - pip install 'strands-robots-sim[isaac]'",
)

# Both the marker AND the explicit env flag are required: booting Isaac Sim's
# SimulationApp is a multi-minute, GPU-only operation, so we never do it unless
# the operator opts in.
_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason=("Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable."),
    ),
]


def _skip_if_isaac_unavailable() -> None:
    """Skip the current test if Isaac Sim's runtime can't boot on this host.

    The plugin may be importable (entry point resolved) while the underlying
    Isaac Sim Kit is missing or the GPU is unusable. ``is_available()`` is the
    plugin's own cheap probe; a False result is an environmental skip, not a
    failure.
    """
    from strands_robots_sim.isaac import IsaacSimulation

    available, msg = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {msg}")


class TestIsaacBackendResolution:
    """The migrated backend must resolve via the documented factory path."""

    def test_create_simulation_isaac_resolves_to_plugin(self):
        """``create_simulation("isaac")`` resolves to the plugin's engine.

        Pins the documented factory UX (``docs/simulation/isaac.md``): the
        ``strands_robots.backends`` entry point makes ``create_simulation
        ("isaac", ...)`` a drop-in peer of ``create_simulation("mujoco")``.
        This runs even before ``SimulationApp`` boots (construction is
        CPU-safe), so it pins the entry-point wiring without needing the Kit.
        """
        from strands_robots_sim.isaac import IsaacSimulation

        from strands_robots.simulation import create_simulation

        sim = create_simulation("isaac", headless=True)
        assert isinstance(sim, IsaacSimulation)

    def test_isaac_is_a_simengine(self):
        """The plugin engine is a ``SimEngine`` so it's drop-in for the agent loop."""
        from strands_robots_sim.isaac import IsaacSimulation

        from strands_robots.simulation.base import SimEngine

        assert issubclass(IsaacSimulation, SimEngine)


class TestIsaacGPUIntegration:
    """End-to-end journeys requiring a real Isaac Sim Kit + GPU."""

    def test_create_world_and_step(self):
        """Create a world, step 100 frames, and read back the step count."""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            result = sim.create_world()
            assert result["status"] == "success", f"create_world: {result}"

            result = sim.step(100)
            assert result["status"] == "success", f"step: {result}"

            state = sim.get_state()
            assert state["status"] == "success", f"get_state: {state}"
            assert state["content"][0].get("json", {}).get("step_count") == 100
        finally:
            sim.destroy()

    def test_render_produces_rgb_image(self):
        """``render()`` must produce a uint8 RGB array."""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True, render_mode="rtx_realtime"))
        try:
            sim.create_world()
            sim.add_camera("cam1", position=[2.0, 2.0, 2.0])
            sim.step(10)

            result = sim.render("cam1")
            assert result["status"] == "success", f"render: {result}"
            assert "rgb" in result
            rgb = result["rgb"]
            assert rgb.shape[2] == 3, f"expected 3 RGB channels, got {rgb.shape}"
            assert rgb.dtype.name == "uint8", f"expected uint8, got {rgb.dtype}"
        finally:
            sim.destroy()

    def test_replicate_fleet_creates_parallel_envs(self):
        """``replicate()`` must create the requested parallel environments."""
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()

        sim = IsaacSimulation(IsaacConfig(num_envs=16, headless=True))
        try:
            sim.create_world()
            sim.add_robot("so100")
            result = sim.replicate(16)
            assert result["status"] == "success", f"replicate: {result}"
            assert "16" in result["content"][0]["text"]
        finally:
            sim.destroy()

    def test_usd_articulation_lifecycle_smoke(self):
        """Boot -> world -> bundled Franka USD -> RTX camera -> step -> teardown.

        Ports the ``examples/libero/run_isaac.py`` lifecycle contract: an
        ``IsaacSimulation`` boots ``SimulationApp``, creates a world, loads a
        bundled Franka USD via ``add_robot(usd_path=...)``, attaches an RTX
        camera, steps physics, and tears down cleanly. Deliberately stops short
        of ``evaluate_benchmark`` (which needs the LIBERO suite importable
        inside Isaac's bundled Python - a separate concern).
        """
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            r = sim.create_world()
            assert r.get("status") == "success", f"create_world: {r}"

            franka_usd = f"{_assets_root_path()}/Isaac/Robots/Franka/franka.usd"
            r = sim.add_robot(name="robot", usd_path=franka_usd)
            assert r.get("status") == "success", f"add_robot: {r}"

            r = sim.add_camera(
                name="image",
                position=[2.0, 0.0, 1.5],
                target=[0.0, 0.0, 0.5],
                fov=60.0,
            )
            assert r.get("status") == "success", f"add_camera: {r}"

            r = sim.step(5)
            assert r.get("status") == "success", f"step: {r}"
        finally:
            sim.destroy()

    def test_send_action_drives_real_usd_articulation(self):
        """``send_action`` must drive a real USD articulation (dict + list forms).

        Regression smoke for the ``set_joint_position_targets`` -> ``apply_action
        (ArticulationAction(...))`` fix: the pre-fix code errored on every
        ``send_action`` because Isaac Sim 6.0's ``SingleArticulation`` has no
        ``set_joint_position_targets``. Loads the bundled Franka, resets + steps
        so it's not an init-timing artefact, then exercises both action forms and
        asserts a success envelope plus a non-empty flat-dict observation whose
        keys are a subset of the articulation's joint names.
        """
        from strands_robots_sim.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            r = sim.create_world()
            assert r.get("status") == "success", f"create_world: {r}"

            franka_usd = f"{_assets_root_path()}/Isaac/Robots/Franka/franka.usd"
            r = sim.add_robot(name="robot", usd_path=franka_usd)
            assert r.get("status") == "success", f"add_robot: {r}"

            # Reset + step so the articulation is fully initialised.
            sim.reset()
            sim.step(5)

            joint_names = sim.robot_joint_names("robot")
            assert isinstance(joint_names, list) and joint_names, f"joint names: {joint_names}"

            # dict form - the documented world-building loop.
            r = sim.send_action({n: 0.0 for n in joint_names}, robot_name="robot")
            assert r.get("status") == "success", f"send_action(dict): {r}"

            # list form - same path, flat array in joint order.
            r = sim.send_action([0.0] * len(joint_names), robot_name="robot")
            assert r.get("status") == "success", f"send_action(list): {r}"

            obs = sim.get_observation(robot_name="robot")
            assert isinstance(obs, dict) and obs, f"get_observation: {obs!r}"
            assert set(obs) <= set(joint_names), f"unexpected obs keys: {sorted(obs)}"
            assert all(isinstance(v, float) for v in obs.values()), f"obs values: {obs}"
        finally:
            sim.destroy()


def _assets_root_path() -> str:
    """Resolve the Isaac Sim bundled-assets root via the modern-then-legacy path.

    Mirrors the example scripts' ``_resolve_robot_asset`` fallback: Isaac Sim 6.0
    exposes ``isaacsim.storage.native.get_assets_root_path``; older builds expose
    ``omni.isaac.nucleus.get_assets_root_path``. Both are imported lazily so
    module import stays CPU-safe.
    """
    try:
        from isaacsim.storage.native import (  # type: ignore[import-not-found]
            get_assets_root_path,
        )
    except ImportError:
        from omni.isaac.nucleus import (  # type: ignore[import-not-found]
            get_assets_root_path,
        )

    assets_root = get_assets_root_path()
    assert assets_root, "get_assets_root_path() returned empty"
    return assets_root
