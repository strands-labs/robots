"""Regression tests: a recorded LeRobotDataset must decode to faithful pixels.

The sibling ``test_recording_codec_opencv`` guards that the per-camera MP4 opens
in OpenCV and yields the right *frame count*. That check reads the raw file with
``cv2.VideoCapture`` - a different decoder from the one a training/replay
pipeline actually uses. LeRobot indexes video frames through its own decode
path (``video_backend`` -> torchcodec/pyav) when a dataset is reopened and
``ds[i]["observation.images.<cam>"]`` is read.

Those two decoders can disagree silently. A torchcodec build whose ABI does not
match the installed torch, or a codec OpenCV cannot decode, yields a file whose
frame count still reads correctly (OpenCV / container header) yet whose decoded
tensor is empty, all-zero, uniform, or channel-swapped - so a training run
consumes garbage pixels while every count-based check passes. This module closes
that gap: it records a scene dominated by a single known colour, reopens the
dataset through LeRobot's own indexing, and asserts the decoded frame is a
correctly shaped, non-degenerate, correctly-ordered RGB tensor. It runs for both
the default H.264 codec and the opt-in AV1 codec (the AV1 path is precisely the
one OpenCV commonly cannot decode but torchcodec must).
"""

import os

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

os.environ.setdefault("MUJOCO_GL", "egl")

# A minimal arm (one actuated hinge, so ``run_policy`` has something to drive)
# in front of a large red panel that fills a close, straight-on camera. A red
# panel makes the decoded frame overwhelmingly red, so a correct RGB decode has
# its red channel as the clear maximum; an RGB<->BGR swap would flip that to the
# blue channel - a robust, resolution-independent channel-order probe.
_ROBOT_XML = """
<mujoco model="fidelity_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.5 0.5 0.5 1"/>
    <geom name="red_panel" type="box" pos="0.30 0 0.10" size="0.02 0.12 0.10" rgba="1 0 0 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.03 0.05" rgba="0.2 0.2 0.2 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
  </actuator>
</mujoco>
"""

_N_STEPS = 12
_CAM = "front"
_KEY = f"observation.images.{_CAM}"


@pytest.fixture
def sim_with_red_panel(tmp_path):
    from strands_robots.simulation import Simulation

    path = tmp_path / "fidelity_arm.xml"
    path.write_text(_ROBOT_XML)

    s = Simulation()
    s.create_world()
    s.add_robot("arm", urdf_path=str(path))
    # Camera looking straight at the red panel from close range: the panel fills
    # the frame so the whole-frame colour signature is unambiguously red.
    s.add_camera(name=_CAM, position=[0.12, 0.0, 0.15], target=[0.30, 0.0, 0.15], width=64, height=64)
    s.step(3)
    yield s
    s.destroy()


def _record(sim, dataset_dir, vcodec=None):
    from strands_robots import MockPolicy

    kwargs = dict(
        repo_id="local/decode_fidelity",
        root=str(dataset_dir),
        fps=30,
        task="decode fidelity",
        overwrite=True,
        cameras=[_CAM],
    )
    if vcodec is not None:
        kwargs["vcodec"] = vcodec
    assert sim.start_recording(**kwargs)["status"] == "success"
    sim.run_policy(robot_name="arm", policy_object=MockPolicy(), n_steps=_N_STEPS)
    assert sim.stop_recording()["status"] == "success"


def _reopen(dataset_dir):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset("local/decode_fidelity", root=str(dataset_dir))


def _decoded_hwc_uint8(ds, idx, key):
    """Index the dataset through LeRobot's own decode path -> HWC uint8 RGB."""
    frame = ds[idx][key]
    arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
    # LeRobot returns image features as CHW float in [0, 1].
    assert arr.ndim == 3 and arr.shape[0] == 3, f"expected (C,H,W) got {arr.shape}"
    return (np.clip(arr.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)


def _assert_faithful_red_frame(hwc, ctx):
    # (1) Non-degenerate: a silent zero/failed decode is an all-black frame
    #     (max ~0); a uniform-garbage decode is a flat field (std ~0). Both are
    #     caught by max + std without penalising a saturated primary colour
    #     (which legitimately zeroes two of its three channels).
    assert hwc.size > 0, f"{ctx}: empty decoded frame"
    assert int(hwc.max()) > 40, f"{ctx}: decoded frame is (near-)black - failed/zero decode (max={hwc.max()})"
    assert float(hwc.std()) > 10.0, f"{ctx}: decoded frame is a flat/uniform field (std={hwc.std():.2f})"

    # (2) Channel order: the red panel dominates the frame, so a faithful RGB
    #     decode has red as the clear maximum channel. An RGB<->BGR swap in the
    #     record/decode round-trip would make blue the maximum instead.
    r, g, b = (float(hwc[..., c].mean()) for c in range(3))
    assert r > g + 15 and r > b + 15, f"{ctx}: not red-dominant (R={r:.1f} G={g:.1f} B={b:.1f}) - channel swap?"


@pytest.mark.parametrize("vcodec", [None, "libsvtav1"], ids=["h264_default", "av1_optin"])
def test_recorded_dataset_decodes_to_faithful_pixels(sim_with_red_panel, tmp_path, vcodec):
    """Reopen a recorded dataset through LeRobot's own indexing and assert the
    decoded frames are non-degenerate, correctly shaped RGB with the expected
    (red-dominant) channel order - for both the default H.264 and opt-in AV1
    codecs.
    """
    _record(sim_with_red_panel, tmp_path / f"ds_{vcodec or 'default'}", vcodec=vcodec)
    ds = _reopen(tmp_path / f"ds_{vcodec or 'default'}")

    assert ds.num_frames == _N_STEPS, f"expected {_N_STEPS} frames, got {ds.num_frames}"
    assert _KEY in ds.features and ds.features[_KEY]["dtype"] == "video"

    # Check the first, a middle, and the last frame (a truncated/partial encode
    # decodes early frames but not late ones).
    for idx in (0, _N_STEPS // 2, _N_STEPS - 1):
        _assert_faithful_red_frame(_decoded_hwc_uint8(ds, idx, _KEY), ctx=f"{vcodec or 'h264'} frame {idx}")
