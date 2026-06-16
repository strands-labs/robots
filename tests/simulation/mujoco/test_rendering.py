"""Tests for multi-camera snapshot + background recording."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("mujoco")

_requires_mujoco = pytest.mark.skipif(
    os.environ.get("CI") == "true" and not os.environ.get("ROBOT_TEST_MUJOCO"),
    reason="requires OpenGL; opt-in via ROBOT_TEST_MUJOCO=1",
)


@_requires_mujoco
def test_render_all_returns_every_camera(tmp_path: Path) -> None:
    """render_all() should return one image block per camera."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    # add 3 cameras
    sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.add_camera("cam_b", position=[0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.add_camera("cam_c", position=[0.0, 0.0, 0.8], target=[0.0, 0.2, 0.05])
    sim.step(n_steps=5)

    r = sim.render_all(width=64, height=48)
    assert r["status"] == "success", r
    image_blocks = [c for c in r["content"] if isinstance(c, dict) and "image" in c]
    # Should include at least the 3 user-added cameras (plus any default)
    assert len(image_blocks) >= 3, f"expected >=3 image blocks, got {len(image_blocks)}"

    # Subset mode
    r2 = sim.render_all(cameras=["cam_a", "cam_c"], width=48, height=32)
    assert r2["status"] == "success"
    imgs = [c for c in r2["content"] if isinstance(c, dict) and "image" in c]
    assert len(imgs) == 2

    sim.destroy()


@_requires_mujoco
def test_start_stop_cameras_recording_writes_one_mp4_per_camera(tmp_path: Path) -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("top", position=[0.0, 0.0, 0.8], target=[0.0, 0.2, 0.05])
    sim.add_camera("side", position=[0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.step(n_steps=5)

    r = sim.start_cameras_recording(
        cameras=["top", "side"],
        output_dir=str(tmp_path),
        fps=20,
        width=64,
        height=48,
        name="integ_test",
    )
    assert r["status"] == "success", r

    # Let it record for ~0.4s of wall time
    time.sleep(0.4)

    status = sim.get_cameras_recording_status()
    assert status["status"] == "success"
    # ASCII-only output contract: active recording is flagged with the
    # "[recording]" marker, never an emoji.
    assert "[recording]" in status["content"][0]["text"]

    stop = sim.stop_cameras_recording()
    assert stop["status"] == "success"

    # Two MP4 files should exist
    files = sorted(tmp_path.glob("*.mp4"))
    names = [f.name for f in files]
    assert any("top" in n for n in names), names
    assert any("side" in n for n in names), names
    for f in files:
        assert f.stat().st_size > 0, f"empty file: {f}"

    sim.destroy()


@_requires_mujoco
def test_stop_without_start_is_idempotent() -> None:
    """T16: idempotent - stop_cameras_recording without a running recording
    returns success with 'Was not recording' instead of erroring."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    r = sim.stop_cameras_recording()
    assert r["status"] == "success"
    assert "Was not recording" in r["content"][0]["text"]
    sim.destroy()


@_requires_mujoco
def test_status_when_idle_is_success() -> None:
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    r = sim.get_cameras_recording_status()
    assert r["status"] == "success"
    # ASCII-only output contract: the idle state is flagged with the
    # "[idle]" marker, never an emoji.
    assert "[idle]" in r["content"][0]["text"]


# Render-time scene_option pass-through (#168 round 9 bug E)


def test_get_viz_option_returns_none_when_unset() -> None:
    """``RenderingMixin._get_viz_option`` returns ``None`` when no
    adapter has populated ``world._backend_state['viz_option']``. This
    is the default path for non-LIBERO sims; ``Renderer.update_scene``
    accepts ``scene_option=None`` as the no-op meaning."""
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    assert sim._get_viz_option() is None
    sim.destroy()


def test_get_viz_option_reads_from_backend_state() -> None:
    """When an adapter (e.g. LiberoAdapter) sets
    ``world._backend_state['viz_option']``, the rendering layer reads it
    via ``_get_viz_option`` and threads it through to
    ``Renderer.update_scene(scene_option=...)``."""
    import mujoco

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[0] = 0  # any non-default value to verify pass-through
    assert sim._world is not None
    sim._world._backend_state["viz_option"] = opt

    retrieved = sim._get_viz_option()
    assert retrieved is opt
    assert int(retrieved.geomgroup[0]) == 0
    sim.destroy()


def test_get_viz_option_handles_missing_backend_state() -> None:
    """Defensive: ``world._backend_state`` not being a dict (e.g. unusual
    test stub) returns None silently rather than raising."""
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    # Force backend_state into an unexpected shape.
    assert sim._world is not None
    sim._world._backend_state = "oops not a dict"  # type: ignore[assignment]
    assert sim._get_viz_option() is None
    sim.destroy()


def test_get_viz_option_handles_missing_world() -> None:
    """``self._world is None`` -> _get_viz_option returns None.
    Reaches this state pre-create_world; defensive guard."""
    from strands_robots.simulation import Simulation

    sim = Simulation()
    # Don't call create_world - self._world stays None.
    assert sim._get_viz_option() is None


# Action-controller hook (#168 round 23 - dispatch action_dict via
# adapter-installed controller in world._backend_state["action_controller"])


def test_get_action_controller_returns_none_when_unset() -> None:
    """``RenderingMixin._get_action_controller`` returns ``None`` when
    no adapter has populated ``world._backend_state['action_controller']``.
    This is the default path for non-LIBERO sims;
    :meth:`_apply_sim_action` falls through to its actuator/joint name
    lookup loop unchanged."""
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    assert sim._get_action_controller() is None
    sim.destroy()


def test_get_action_controller_reads_from_backend_state() -> None:
    """When an adapter sets ``world._backend_state['action_controller']``,
    the rendering layer reads it via ``_get_action_controller`` and
    dispatches to ``controller.apply(...)`` instead of the name-lookup
    loop."""
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sentinel_controller = object()
    assert sim._world is not None
    sim._world._backend_state["action_controller"] = sentinel_controller
    assert sim._get_action_controller() is sentinel_controller
    sim.destroy()


def test_get_action_controller_handles_missing_backend_state() -> None:
    """Defensive: ``world._backend_state`` not being a dict returns
    None silently rather than raising."""
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    assert sim._world is not None
    sim._world._backend_state = "oops not a dict"  # type: ignore[assignment]
    assert sim._get_action_controller() is None
    sim.destroy()


def test_get_action_controller_handles_missing_world() -> None:
    """``self._world is None`` -> _get_action_controller returns None."""
    from strands_robots.simulation import Simulation

    sim = Simulation()
    assert sim._get_action_controller() is None


@_requires_mujoco
def test_apply_sim_action_dispatches_to_controller_when_installed() -> None:
    """When an action_controller is installed, ``_apply_sim_action``
    dispatches to ``controller.apply(action_dict, model, data, robot_name)``
    instead of the actuator/joint name lookup loop. Pin for #168
    round-23 contract: GR00T's task-space action keys (``x``, ``y``,
    ``z``, ``roll``, ``pitch``, ``yaw``, ``gripper``) would otherwise
    silently no-op because they don't match any actuator name."""
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "glfw")

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])

    apply_calls: list = []

    class _CountingController:
        def apply(self, action_dict, model, data, robot_name):
            apply_calls.append((dict(action_dict), robot_name))
            # Don't write data.ctrl - just record the call.

    assert sim._world is not None
    sim._world._backend_state["action_controller"] = _CountingController()

    # Send an action dict that wouldn't match any actuator in the so101
    # model - the name-lookup loop would silently drop these. The
    # controller should still get called.
    sim.send_action({"x": 0.5, "gripper": 0.7}, robot_name="arm")
    assert len(apply_calls) == 1
    action, robot_name = apply_calls[0]
    assert action == {"x": 0.5, "gripper": 0.7}
    assert robot_name == "arm"
    sim.destroy()


@_requires_mujoco
def test_apply_sim_action_falls_back_when_controller_raises() -> None:
    """If the action_controller's ``apply`` raises, ``_apply_sim_action``
    logs a WARNING and falls through to the name-lookup loop. Pin for
    the controller-failure recovery path so a transient OSC controller
    crash doesn't silently zero the action stream."""
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "glfw")

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])

    class _BoomController:
        def apply(self, action_dict, model, data, robot_name):
            raise RuntimeError("simulated controller failure")

    assert sim._world is not None
    sim._world._backend_state["action_controller"] = _BoomController()

    # The name-lookup fallback fires; a recognized actuator key gets
    # applied. Use a likely-existing so101 actuator name.
    actuator_names = []
    import mujoco

    for i in range(int(sim._world._model.nu)):
        n = mujoco.mj_id2name(sim._world._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if n:
            actuator_names.append(n)
    if not actuator_names:
        pytest.skip("no actuators in so101 model; can't test fallback")

    # Use the FIRST actuator's bare name (strip namespace prefix if any).
    raw = actuator_names[0]
    pfx = "arm/"
    bare = raw[len(pfx) :] if raw.startswith(pfx) else raw

    sim.send_action({bare: 0.5}, robot_name="arm")
    # Look up the corresponding ctrl[i].
    ctrl_id = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_ACTUATOR, raw)
    assert ctrl_id >= 0
    # The fallback's name-lookup loop wrote 0.5 to data.ctrl[ctrl_id].
    assert sim._world._data.ctrl[ctrl_id] == 0.5

    sim.destroy()


def test_apply_sim_action_no_controller_uses_name_lookup() -> None:
    """Without an action_controller installed, ``_apply_sim_action``
    uses the name-lookup loop (the pre-round-23 default behaviour).
    Pin so non-LIBERO callers and existing tests see zero behaviour
    change."""
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "glfw")

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])

    assert sim._world is not None
    # No action_controller installed.
    assert sim._get_action_controller() is None

    # Random unmatched key - falls through silently (the round-23
    # diagnostic pattern: GR00T-shaped keys with no name match).
    sim.send_action({"this_key_does_not_match_any_actuator": 1.0}, robot_name="arm")
    # data.ctrl all zero.
    import numpy as np

    assert np.all(sim._world._data.ctrl == 0)

    sim.destroy()


@_requires_mujoco
def test_render_passes_scene_option_to_renderer(tmp_path: Path) -> None:
    """End-to-end: ``Simulation.render(camera_name=...)`` reads
    viz_option from backend_state and threads it through to the
    underlying ``Renderer.update_scene`` so the rendered frame
    reflects the option (e.g. group=0 geoms hidden).

    Pin for #168 round 9: the round-9 fix moves collision-geom hiding
    from MJCF rgba edits to renderer-level mjvOption. Without
    backend_state -> Renderer threading, viz_option would be
    populated by adapters but ignored at render time."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])

    # Default render (no viz_option) - just verify it returns success.
    default = sim.render(camera_name="cam", width=64, height=64)
    assert default["status"] == "success"

    # Install a viz_option that turns OFF every visible group. The
    # rendered frame should be uniform / nearly-empty (no geoms drawn).
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    for g in range(6):
        opt.geomgroup[g] = 0  # hide ALL geom groups
    for sg in range(6):
        opt.sitegroup[sg] = 0
    assert sim._world is not None
    sim._world._backend_state["viz_option"] = opt

    masked = sim.render(camera_name="cam", width=64, height=64)
    assert masked["status"] == "success"
    # The rendered image should have very low pixel variance now that
    # everything is hidden - just background sky / floor.
    masked_var = next(
        c["json"]["pixel_variance"]
        for c in masked["content"]
        if isinstance(c, dict) and "json" in c and "pixel_variance" in c["json"]
    )
    default_var = next(
        c["json"]["pixel_variance"]
        for c in default["content"]
        if isinstance(c, dict) and "json" in c and "pixel_variance" in c["json"]
    )
    # Hiding everything should significantly reduce variance.
    assert masked_var < default_var, (
        f"viz_option not honoured: masked variance {masked_var} should be < default {default_var}"
    )

    sim.destroy()
    sim.destroy()


# Recorder thread warms up its GL context adaptively before the
# capture loop (#168 round-21: replaces the round-20 fixed 2-pass
# warmup with warmup-until-warm. Round-20 verification showed image
# channel needed ~15 frames to clear vs wrist's 3 frames - per-camera
# warmup latency varies. Round 21 keeps rendering each camera until
# its output passes the col-std threshold, capped at 30 attempts.)


@_requires_mujoco
def test_recorder_thread_warms_up_until_each_camera_clears() -> None:
    """The recorder thread renders each camera adaptively until it
    produces non-gradient output (col-std > threshold), then starts
    the timing loop.

    Round-21 contract: warmup loop iterates up to MAX_WARMUP_RENDERS
    times. Each iteration, for every still-cold camera, render once
    and check ``arr.std(axis=0).mean()`` against the threshold (5.0).
    Stop iterating when all cameras are warm or cap is hit.

    Round-20's fixed 2-pass approach was insufficient for the FIRST
    camera in multi-camera recordings - image channel stayed cold
    for ~15 frames while wrist cleared at frame 3. The shared
    ``mujoco.Renderer`` rebinds per-camera state and the per-camera
    warmup latency varies (likely GPU command-buffer flush
    ordering).

    Test mechanism: mock ``threading.Thread`` to capture ``_loop``
    without starting it, mock ``sim.render`` to return a stub frame
    with ``arr.std(axis=0).mean() > 5`` immediately. Warmup should
    fire exactly once per camera (then mark warm and exit).
    """
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "glfw")

    import io

    import numpy as np
    from PIL import Image

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam_a", position=[0.5, 0, 0.5], target=[0, 0, 0])
    sim.add_camera("cam_b", position=[-0.5, 0, 0.5], target=[0, 0, 0])

    # Stub frame with non-zero column variance (col-std > 5).
    # Construct a checkerboard-like pattern.
    arr = np.zeros((48, 64, 3), dtype=np.uint8)
    arr[:24, :, :] = 255  # top half white, bottom half black -> high col std
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    render_calls: list[str] = []

    def stub_render(camera_name: str, width=None, height=None) -> dict:
        render_calls.append(camera_name)
        return {
            "status": "success",
            "content": [
                {"text": camera_name},
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": png_bytes, "media_type": "image/png"},
                    }
                },
                {"json": {"pixel_variance": 100.0, "pixel_mean": 128.0}},
            ],
        }

    captured_targets: list = []

    class _CaptureThread:
        def __init__(self, target=None, daemon=None) -> None:
            self.target = target
            self.daemon = daemon

        def start(self) -> None:
            captured_targets.append(self.target)

        def is_alive(self) -> bool:
            return False

        def join(self, timeout=None) -> None:
            pass

    with patch.object(sim, "render", side_effect=stub_render):
        with patch("threading.Thread", _CaptureThread):
            r = sim.start_cameras_recording(
                cameras=["cam_a", "cam_b"],
                output_dir="/tmp",
                fps=20,
                width=64,
                height=48,
                name="warmup_test",
            )
            assert r["status"] == "success", r
            assert len(captured_targets) == 1
            sim._cams_rec_state["running"] = False
            captured_targets[0]()  # invoke _loop synchronously

    # With non-gradient stub frames returned immediately, both cameras
    # warm on attempt 1: 2 renders total (one per camera).
    assert render_calls == ["cam_a", "cam_b"], f"expected one render per camera (warm immediately), got {render_calls}"

    sim.destroy()


@_requires_mujoco
def test_recorder_thread_warmup_continues_when_camera_stays_cold() -> None:
    """If a camera consistently returns gradient frames, warmup retries
    up to MAX_WARMUP_RENDERS (30) times before giving up. Last
    iteration logs WARNING about cameras still cold.

    Pin for #168 round-21 contract: the warmup is bounded; cap is
    30 attempts so the worst-case overhead is ~1 second at 30 fps.
    Common case is much faster (1-3 attempts per camera)."""
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "glfw")

    import io

    import numpy as np
    from PIL import Image

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam", position=[0.5, 0, 0.5], target=[0, 0, 0])

    # Stub: ALWAYS gradient (uniform color, col-std ~0)
    gradient = np.full((48, 64, 3), 128, dtype=np.uint8)
    pil = Image.fromarray(gradient)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    render_calls: list[str] = []

    def gradient_render(camera_name: str, width=None, height=None) -> dict:
        render_calls.append(camera_name)
        return {
            "status": "success",
            "content": [
                {"text": camera_name},
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": png_bytes, "media_type": "image/png"},
                    }
                },
                {"json": {"pixel_variance": 0.0, "pixel_mean": 128.0}},
            ],
        }

    captured_targets: list = []

    class _CaptureThread:
        def __init__(self, target=None, daemon=None) -> None:
            self.target = target

        def start(self) -> None:
            captured_targets.append(self.target)

        def is_alive(self) -> bool:
            return False

        def join(self, timeout=None) -> None:
            pass

    with patch.object(sim, "render", side_effect=gradient_render):
        with patch("threading.Thread", _CaptureThread):
            sim.start_cameras_recording(
                cameras=["cam"],
                output_dir="/tmp",
                fps=20,
                width=64,
                height=48,
                name="cold_test",
            )
            sim._cams_rec_state["running"] = False
            captured_targets[0]()

    # Capped at 30 attempts.
    assert len(render_calls) == 30, f"expected 30 warmup attempts (capped), got {len(render_calls)}"

    sim.destroy()


@_requires_mujoco
def test_recorder_thread_warmup_failure_does_not_abort() -> None:
    """If the thread-side warmup render raises, the timing loop
    still starts (and accumulates ``state['errors'][cam]`` per the
    standard error-tracking path).

    Required because warmup failure shouldn't crash the recorder
    thread; persistent failures will surface via
    :meth:`get_cameras_recording_status`."""
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "glfw")

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam", position=[0.5, 0, 0.5], target=[0, 0, 0])

    def boom(camera_name: str, width=None, height=None):
        raise RuntimeError(f"simulated warmup failure on {camera_name}")

    captured_targets: list = []

    class _CaptureThread:
        def __init__(self, target=None, daemon=None) -> None:
            self.target = target

        def start(self) -> None:
            captured_targets.append(self.target)

        def is_alive(self) -> bool:
            return False

    with patch.object(sim, "render", side_effect=boom):
        with patch("threading.Thread", _CaptureThread):
            r = sim.start_cameras_recording(
                cameras=["cam"],
                output_dir="/tmp",
                fps=20,
                width=64,
                height=48,
                name="warmup_fail_test",
            )
            assert r["status"] == "success", r
            assert len(captured_targets) == 1
            # Stop the timing loop before invoking _loop.
            sim._cams_rec_state["running"] = False
            # Must not raise even though warmup renders all raise.
            captured_targets[0]()

    sim.destroy()


@_requires_mujoco
def test_recorder_first_frame_is_real_geometry(tmp_path: Path) -> None:
    """End-to-end: the FIRST frame written to the MP4 is real geometry
    (col-std > 30), not the skybox-only gradient (col-std ~0.6).

    Pin for #168 round-11 bug D: pre-fix, the recorder thread's first
    captured frame was a cold-start gradient because the renderer's
    scene buffer hadn't been populated. The synchronous warmup before
    thread launch ensures the first thread-side render() lands on the
    warm path.

    Gated behind ``_requires_mujoco`` because it needs a real GL
    context to actually render meaningful pixels."""
    os.environ.setdefault("MUJOCO_GL", "glfw")

    import imageio.v2 as imageio
    import numpy as np

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam", position=[0.4, -0.4, 0.5], target=[0.0, 0.0, 0.1])
    sim.step(n_steps=5)

    # Brand-new recorder; first frame must be warm.
    r = sim.start_cameras_recording(
        cameras=["cam"],
        output_dir=str(tmp_path),
        fps=20,
        width=64,
        height=48,
        name="first_frame_test",
    )
    assert r["status"] == "success", r

    time.sleep(0.2)
    sim.stop_cameras_recording()

    # Decode the MP4 and inspect the first frame's column stddev.
    mp4_files = list(tmp_path.glob("*.mp4"))
    assert mp4_files, "expected at least one mp4 file"
    reader = imageio.get_reader(str(mp4_files[0]))
    try:
        frames: list = []
        for frame in reader.iter_data():  # type: ignore[attr-defined]
            frames.append(frame)
    finally:
        reader.close()
    assert len(frames) >= 1
    first = np.asarray(frames[0])
    # Real-geometry frames have col-std around 30+; cold-start gradient
    # frames have col-std around 0.6 (dominated by the smooth horizontal
    # skybox blue->grey gradient).
    col_std = float(first.std(axis=0).mean())
    assert col_std > 5.0, (
        f"first frame appears to be cold-start gradient, col_std={col_std:.2f}; "
        f"expected real geometry (col_std>5). Frame mean RGB: "
        f"({first[..., 0].mean():.1f}, {first[..., 1].mean():.1f}, {first[..., 2].mean():.1f})"
    )

    sim.destroy()
