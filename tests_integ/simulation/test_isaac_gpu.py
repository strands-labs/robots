"""GPU integration tests for the Isaac Sim simulation backend.

Ported from ``strands-robots-sim`` (``strands_robots_sim/isaac/tests/
test_gpu_integ.py``) as part of the Isaac backend migration
(`#1144 <https://github.com/strands-labs/robots/issues/1144>`_ EPIC, child
`#1145 <https://github.com/strands-labs/robots/issues/1145>`_).

The Isaac Sim backend is now a **vendored in-tree built-in** living at
``strands_robots.simulation.isaac`` (it previously shipped out-of-tree in the
sibling ``strands-robots-sim`` plugin under the ``strands_robots.backends``
entry-point group). These tests exercise the backend through both the
documented factory path (``create_simulation("isaac", ...)``) and the direct
constructor (``IsaacSimulation(IsaacConfig(...))``), so the backend contract is
pinned regardless of which entry point a caller uses.

Requirements:
  - NVIDIA GPU with CUDA
  - Isaac Sim 6.0+ installed (Python 3.12), installed out-of-band via
    Omniverse / Isaac Lab / the NGC docker image (see
    ``strands_robots.simulation.isaac._install``)
  - ``pip install 'strands-robots[sim-isaac]'`` (provides the usd-core/imageio
    helpers; Isaac Sim itself is not pip-installable)
  - Environment variable: ``STRANDS_GPU_TEST=1``

Gated behind the ``gpu`` marker AND ``STRANDS_GPU_TEST=1`` so a CI run that
happens to have Isaac Sim installed without a GPU never boots ``SimulationApp``
and times out. ``importorskip`` emits a clean SKIPPED line (not an opaque
collection-time ImportError) when the backend deps are absent.

Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ \\
        tests_integ/simulation/test_isaac_gpu.py -m gpu -v

These assert only user-visible behaviour (status envelopes, observation
shapes, RGB frame dtype/channels) - never internal simulation state - per the
"test behavior, not implementation" rule.
"""

from __future__ import annotations

import os

import pytest

# The isaac subpackage import itself is CPU-safe (all heavy omni/isaacsim
# imports are lazy, deferred to create_world()); importorskip guards against a
# broken/partial install rather than the Isaac Kit runtime.
pytest.importorskip("strands_robots.simulation.isaac")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]


def _skip_if_isaac_unavailable() -> None:
    """Skip the current test if Isaac Sim's runtime can't boot on this host.

    The subpackage may be importable (CPU-safe) while the underlying Isaac Sim
    Kit is missing or the GPU is unusable. ``is_available()`` is the backend's
    own cheap probe; a False result is an environmental skip, not a failure.
    """
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


class TestIsaacBackendResolution:
    """The vendored backend must resolve via the documented factory path."""

    def test_create_simulation_isaac_resolves_to_builtin(self):
        """``create_simulation("isaac")`` resolves to the in-tree engine.

        Pins the documented factory UX: ``isaac`` is now a first-class
        built-in peer of ``mujoco`` / ``newton`` (no plugin install required).
        Construction is CPU-safe (the heavy omni/isaacsim import is deferred to
        ``create_world()``), so this runs even before ``SimulationApp`` boots.
        """
        from strands_robots.simulation import create_simulation
        from strands_robots.simulation.isaac import IsaacSimulation

        sim = create_simulation("isaac", num_envs=1, headless=True)
        assert isinstance(sim, IsaacSimulation)

    def test_isaac_is_a_simengine(self):
        """The engine is a ``SimEngine`` so it's drop-in for the agent loop."""
        from strands_robots.simulation.base import SimEngine
        from strands_robots.simulation.isaac import IsaacSimulation

        assert issubclass(IsaacSimulation, SimEngine)


class TestIsaacGPUIntegration:
    """End-to-end journeys requiring a real Isaac Sim Kit + GPU."""

    def test_create_world_and_step(self):
        """Create a world, step frames, and read back the step count."""
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            r = sim.create_world()
            assert r["status"] == "success", f"create_world: {r}"

            r = sim.step(100)
            assert r["status"] == "success", f"step: {r}"

            r = sim.get_state()
            assert r["status"] == "success", f"get_state: {r}"
            state = r["content"][0]["json"]
            assert state["step_count"] == 100
        finally:
            sim.destroy()

    def test_render_produces_rgb_image(self):
        """``render()`` returns a contract-clean envelope; ``_render_frame`` a
        uint8 RGB array.

        The public ``render()`` returns only ``{status, content}`` (raw pixels
        moved off the top level into the internal ``_render_frame`` helper +
        a content PNG block, per the tool-result contract). This test pins
        both surfaces.
        """
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=False, render_mode="rtx_realtime"))
        try:
            r = sim.create_world()
            assert r["status"] == "success", f"create_world: {r}"

            r = sim.add_camera("cam1")
            assert r["status"] == "success", f"add_camera: {r}"

            sim.step(2)

            r = sim.render("cam1")
            assert r["status"] == "success", f"render: {r}"
            # Contract-clean envelope: only status + content, no top-level rgb.
            assert set(r.keys()) == {"status", "content"}
            assert "rgb" not in r

            # Raw pixels come from the internal helper, not the tool-result dict.
            rgb, _depth, _meta = sim._render_frame("cam1")
            assert rgb is not None, "expected a non-None RGB frame"
            assert rgb.shape[-1] == 3, f"expected 3 RGB channels, got {rgb.shape[-1]}"
            assert rgb.dtype.name == "uint8", f"expected uint8, got {rgb.dtype}"
        finally:
            sim.destroy()

    def test_replicate_fleet_creates_parallel_envs(self):
        """``replicate()`` must create the requested parallel environments."""
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        sim = IsaacSimulation(IsaacConfig(num_envs=16, headless=True))
        try:
            r = sim.create_world()
            assert r["status"] == "success", f"create_world: {r}"

            r = sim.add_robot("so100")
            assert r["status"] == "success", f"add_robot: {r}"

            r = sim.replicate(16)
            assert r["status"] == "success", f"replicate: {r}"
            assert "16" in r["content"][0]["text"]
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
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        assets_root = _assets_root_path()
        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=False, render_mode="rtx_realtime"))
        try:
            r = sim.create_world()
            assert r["status"] == "success", f"create_world: {r}"

            usd_path = f"{assets_root}/Isaac/Robots/Franka/franka.usd"
            r = sim.add_robot("robot", usd_path=usd_path)
            assert r["status"] == "success", f"add_robot: {r}"

            r = sim.add_camera("cam1")
            assert r["status"] == "success", f"add_camera: {r}"

            r = sim.step(10)
            assert r["status"] == "success", f"step: {r}"
        finally:
            sim.destroy()

    def test_send_action_drives_real_usd_articulation(self):
        """``send_action`` must drive a real USD articulation (dict + list forms).

        Regression smoke for the ``set_joint_position_targets`` ->
        ``apply_action(ArticulationAction(...))`` fix: the pre-fix code errored
        on every ``send_action`` because Isaac Sim 6.0's ``SingleArticulation``
        has no ``set_joint_position_targets``. Loads the bundled Franka, resets
        + steps so it's not an init-timing artefact, then exercises both action
        forms and asserts a success envelope plus a non-empty flat-dict
        observation whose keys are a subset of the articulation's joint names.
        """
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        assets_root = _assets_root_path()
        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            r = sim.create_world()
            assert r["status"] == "success", f"create_world: {r}"

            usd_path = f"{assets_root}/Isaac/Robots/Franka/franka.usd"
            r = sim.add_robot("robot", usd_path=usd_path)
            assert r["status"] == "success", f"add_robot: {r}"

            sim.reset()
            sim.step(2)

            joint_names = sim.robot_joint_names("robot")
            assert isinstance(joint_names, list), f"joint names: {joint_names}"
            assert len(joint_names) > 0, f"joint names: {joint_names}"

            action_dict = {name: 0.0 for name in joint_names}
            r = sim.send_action(action_dict, robot_name="robot")
            assert r["status"] == "success", f"send_action(dict): {r}"

            action_list = [0.0] * len(joint_names)
            r = sim.send_action(action_list, robot_name="robot")
            assert r["status"] == "success", f"send_action(list): {r}"

            obs = sim.get_observation("robot")
            assert isinstance(obs, dict), f"get_observation: {obs}"
            assert len(obs) > 0, f"get_observation: {obs}"
            assert set(obs.keys()).issubset(set(joint_names)), f"unexpected obs keys: {set(obs.keys())}"
            assert all(isinstance(v, float) for v in obs.values()), f"obs values: {obs}"
        finally:
            sim.destroy()


def _assets_root_path() -> str:
    """Resolve the Isaac Sim bundled-assets root via the modern-then-legacy path.

    Mirrors the example scripts' ``_resolve_robot_asset`` fallback: Isaac Sim
    6.0 exposes ``isaacsim.storage.native.get_assets_root_path``; older builds
    expose ``omni.isaac.nucleus.get_assets_root_path``. Both are imported
    lazily so module import stays CPU-safe.
    """
    try:
        from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]
    except ImportError:
        from omni.isaac.nucleus import get_assets_root_path  # type: ignore[import-not-found]

    assets_root = get_assets_root_path()
    assert assets_root, "get_assets_root_path() returned empty"
    return assets_root
