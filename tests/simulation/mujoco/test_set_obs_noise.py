"""MuJoCo backend implements ``set_obs_noise`` (additive sensor-noise contract).

``set_obs_noise`` is a declared ``SimEngine`` method - Newton implements it fully
(joint-position/velocity noise on observations + camera-frame jitter), but the
default MuJoCo backend historically inherited the base ``NotImplementedError``.
That left sim-to-real robustness code that runs on Newton crashing on MuJoCo,
the reference backend. These tests pin the MuJoCo implementation and its
parity with Newton's contract:

  * ``set_obs_noise(...)`` succeeds and is advertised in ``describe()``.
  * joint-position and ``.vel`` noise perturb ``get_observation`` (pos + vel).
  * position + velocity noise perturb ``get_robot_state``.
  * a seed makes the noise stream reproducible.
  * negative / non-finite / non-numeric values are rejected with ``status=error``.
  * disabling (all-zero std) restores noise-free observations, and the default
    (never-configured) path is an exact no-op.
  * ``camera_jitter_px`` shifts rendered frames.

Every test that calls ``set_obs_noise`` fails pre-fix with ``NotImplementedError``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No OpenGL context available (headless without EGL/OSMesa)",
)


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_obs_noise", mesh=False)
    assert s.create_world(gravity=[0, 0, -9.81])["status"] == "success"
    assert s.add_robot("so101")["status"] == "success"
    yield s
    s.cleanup()


def _pos_keys(obs: dict) -> list[str]:
    return [k for k, v in obs.items() if isinstance(v, float) and not k.endswith(".vel")]


def _vel_keys(obs: dict) -> list[str]:
    return [k for k in obs if k.endswith(".vel")]


def test_set_obs_noise_succeeds_and_is_discoverable(sim):
    """The MuJoCo backend implements set_obs_noise (pre-fix: NotImplementedError)."""
    result = sim.set_obs_noise(joint_pos_std=0.01)
    assert result["status"] == "success", result
    assert "set_obs_noise" in sim.describe()["methods"]


def test_joint_position_and_velocity_noise_perturbs_observation(sim):
    """joint_pos_std perturbs positions and joint_vel_std perturbs `.vel`."""
    base = sim.get_observation(skip_images=True)
    pos_keys, vel_keys = _pos_keys(base), _vel_keys(base)
    assert pos_keys and vel_keys, "so101 should expose position + velocity keys"

    assert sim.set_obs_noise(joint_pos_std=0.05, joint_vel_std=0.1, seed=0)["status"] == "success"
    noisy = sim.get_observation(skip_images=True)

    assert max(abs(noisy[k] - base[k]) for k in pos_keys) > 0, "position noise had no effect"
    assert max(abs(noisy[k] - base[k]) for k in vel_keys) > 0, "velocity noise had no effect"


def test_get_robot_state_is_noised(sim):
    """set_obs_noise perturbs get_robot_state position + velocity fields."""
    clean = sim.get_robot_state()["content"][1]["json"]["state"]
    assert sim.set_obs_noise(joint_pos_std=0.05, joint_vel_std=0.1, seed=0)["status"] == "success"
    noisy = sim.get_robot_state()["content"][1]["json"]["state"]

    j = next(iter(clean))
    assert abs(noisy[j]["position"] - clean[j]["position"]) > 0
    assert abs(noisy[j]["velocity"] - clean[j]["velocity"]) > 0


def test_seeded_noise_is_reproducible(sim):
    """Re-seeding with the same seed reproduces the identical noise stream."""
    keys = _pos_keys(sim.get_observation(skip_images=True))
    sim.set_obs_noise(joint_pos_std=0.05, seed=42)
    a = sim.get_observation(skip_images=True)
    sim.set_obs_noise(joint_pos_std=0.05, seed=42)
    b = sim.get_observation(skip_images=True)
    assert all(abs(a[k] - b[k]) < 1e-12 for k in keys)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"joint_pos_std": -0.1},
        {"joint_vel_std": -1.0},
        {"camera_jitter_px": float("nan")},
        {"joint_pos_std": float("inf")},
        {"joint_pos_std": "fast"},
    ],
)
def test_invalid_values_rejected(sim, kwargs):
    """Negative / non-finite / non-numeric values return status=error."""
    assert sim.set_obs_noise(**kwargs)["status"] == "error"


def test_default_path_is_noise_free_and_disable_restores_it(sim):
    """Unconfigured get_observation is an exact no-op; all-zero std disables noise."""
    base = sim.get_observation(skip_images=True)
    # never configured -> repeated observation is byte-identical
    assert sim.get_observation(skip_images=True) == base

    sim.set_obs_noise(joint_pos_std=0.05, seed=0)
    assert sim.get_observation(skip_images=True) != base

    sim.set_obs_noise(joint_pos_std=0.0, joint_vel_std=0.0, camera_jitter_px=0.0)
    restored = sim.get_observation(skip_images=True)
    pos_keys = _pos_keys(base)
    assert max(abs(restored[k] - base[k]) for k in pos_keys) < 1e-9


def test_maybe_jitter_frame_shifts_synthetic_frame(sim):
    """The jitter helper rolls a non-uniform frame by an integer offset (no GL)."""
    frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    # seed=0 draws a non-zero (dy, dx); (a zero draw is a legitimate but
    # uninformative RNG outcome, so pin a seed that actually shifts).
    sim.set_obs_noise(camera_jitter_px=8, seed=0)
    jittered = sim._maybe_jitter_frame(frame)
    assert jittered.shape == frame.shape
    assert not np.array_equal(jittered, frame), "jitter helper did not shift the frame"

    # disabled -> exact identity (same object returned)
    sim.set_obs_noise(camera_jitter_px=0.0)
    assert sim._maybe_jitter_frame(frame) is frame


@requires_gl
def test_camera_jitter_shifts_rendered_frame(sim):
    """camera_jitter_px shifts the rendered frame vs the noise-free render."""
    assert (
        sim.add_camera(name="cam1", position=[0.5, 0.0, 0.3], target=[0.0, 0.0, 0.1], width=128, height=96)["status"]
        == "success"
    )

    def _png(result: dict) -> np.ndarray:
        assert result["status"] == "success", result
        for block in result["content"]:
            if "image" in block:
                return np.array(Image.open(io.BytesIO(block["image"]["source"]["bytes"])))
        raise AssertionError("no image block")

    from PIL import Image

    sim.set_obs_noise(camera_jitter_px=0.0)
    clean = _png(sim.render(camera_name="cam1", width=128, height=96))
    sim.set_obs_noise(camera_jitter_px=8, seed=0)
    jittered = _png(sim.render(camera_name="cam1", width=128, height=96))
    assert not np.array_equal(clean, jittered), "camera jitter had no effect on the render"


def test_apply_obs_noise_jitters_frames_and_passes_through_floating_base_lists():
    """`_apply_obs_noise` jitters ndarray frames and leaves floating-base lists intact.

    `get_observation` returns a heterogeneous dict: scalar joint values, camera
    frames as ndarrays, and (for floating-base robots) `base_quat` / `base_ang_vel`
    list values. Existing coverage only ever exercises `_apply_obs_noise` with
    `skip_images=True` on the fixed-base so101, so the ndarray-jitter branch and
    the floating-base list passthrough documented in the method contract were
    never driven. Build the heterogeneous obs directly (no GL required) and pin
    both behaviours: the frame is shifted, the quaternion/angular-velocity lists
    are returned untouched (a quaternion would need renormalisation - out of
    scope for additive scalar noise).
    """
    s = Simulation(tool_name="test_obs_noise_apply", mesh=False)
    try:
        # camera_jitter_px is the only configured noise, so the RNG is consumed
        # solely by the frame jitter -> seed=0 reproduces the shifting draw the
        # standalone jitter test relies on.
        assert s.set_obs_noise(camera_jitter_px=8, seed=0)["status"] == "success"

        frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        base_quat = [1.0, 0.0, 0.0, 0.0]
        base_ang_vel = [0.1, 0.2, 0.3]
        obs = {
            "shoulder_pan": 0.5,
            "shoulder_pan.vel": 0.1,
            "front_cam": frame,
            "base_quat": base_quat,
            "base_ang_vel": base_ang_vel,
        }

        out = s._apply_obs_noise(obs)

        # ndarray frame routed through the jitter path (shape preserved, shifted).
        assert out["front_cam"].shape == frame.shape
        assert not np.array_equal(out["front_cam"], frame), "camera frame was not jittered"
        # Floating-base list signals pass through unchanged (same object).
        assert out["base_quat"] is base_quat
        assert out["base_ang_vel"] is base_ang_vel
    finally:
        s.cleanup()


def test_sub_pixel_camera_jitter_is_a_noop(sim):
    """A sub-pixel `camera_jitter_px` (0 < px < 1) rounds down to no shift.

    `np.roll` can only shift by whole pixels, so a fractional jitter setting
    floors to a zero max-shift and returns the frame unchanged rather than
    raising - a legitimate, if uninformative, configuration.
    """
    frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    assert sim.set_obs_noise(camera_jitter_px=0.5, seed=0)["status"] == "success"
    assert sim._maybe_jitter_frame(frame) is frame
