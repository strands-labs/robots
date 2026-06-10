"""Unit tests for ``RecordingMixin._verify_resume_schema``.

The schema-verification helper is a pure function of its arguments (it reads no
instance state), so these tests bind it to a dummy ``self`` and run without
mujoco or lerobot installed. They pin the #366 follow-up: resume() must reject a
scene whose schema diverges from the existing on-disk dataset rather than
deferring to a cryptic per-feature shape error on the next add_frame.
"""

import pytest

from strands_robots.simulation.mujoco.recording import RecordingMixin

verify = RecordingMixin._verify_resume_schema


class _FakeRecorder:
    def __init__(self, features):
        self.dataset = type("_DS", (), {"features": features})()


def _features(joint_names, cams):
    """Build a minimal on-disk feature dict: cams maps name -> (h, w)."""
    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(joint_names),), "names": list(joint_names)},
    }
    for name, (h, w) in cams.items():
        feats[f"observation.images.{name}"] = {"dtype": "video", "shape": (3, h, w)}
    return feats


def test_resume_schema_matching_scene_passes():
    rec = _FakeRecorder(_features(["shoulder_pan", "elbow"], {"front": (480, 640)}))
    # No raise -> schema matches.
    verify(None, rec, ["shoulder_pan", "elbow"], ["front"], {"front": (480, 640)})


def test_resume_schema_extra_joint_raises():
    rec = _FakeRecorder(_features(["shoulder_pan"], {}))
    with pytest.raises(ValueError, match="observation.state joints differ"):
        verify(None, rec, ["shoulder_pan", "elbow"], [], {})


def test_resume_schema_camera_resolution_mismatch_raises():
    rec = _FakeRecorder(_features(["j"], {"front": (480, 640)}))
    with pytest.raises(ValueError, match="resolution differs"):
        verify(None, rec, ["j"], ["front"], {"front": (256, 256)})


def test_resume_schema_new_camera_in_scene_raises():
    rec = _FakeRecorder(_features(["j"], {"front": (480, 640)}))
    with pytest.raises(ValueError, match="not in the on-disk schema"):
        verify(None, rec, ["j"], ["front", "wrist"], {"front": (480, 640), "wrist": (480, 640)})


def test_resume_schema_dropped_camera_raises():
    rec = _FakeRecorder(_features(["j"], {"front": (480, 640), "wrist": (480, 640)}))
    with pytest.raises(ValueError, match="not in the current scene"):
        verify(None, rec, ["j"], ["front"], {"front": (480, 640)})


def test_resume_schema_no_features_skips_silently():
    """An unexpected LeRobot layout (no .features) must not block a valid resume."""
    rec = type("_R", (), {"dataset": type("_DS", (), {})()})()
    verify(None, rec, ["j"], [], {})  # no raise


def test_resume_schema_error_message_is_ascii():
    rec = _FakeRecorder(_features(["shoulder_pan"], {}))
    with pytest.raises(ValueError) as exc:
        verify(None, rec, ["shoulder_pan", "elbow"], [], {})
    str(exc.value).encode("ascii")  # raises if any non-ASCII glyph leaked
