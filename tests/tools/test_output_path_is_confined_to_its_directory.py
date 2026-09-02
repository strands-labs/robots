"""A caller-named file must land in the directory that was validated.

``validate_save_path`` is the guard every filesystem-writing tool here runs on the
directory it was told to write into. It rejects a ``..`` component explicitly, and
resolves the result to check it does not land in a protected system directory. What
it cannot do is validate the *file*, because the file name is composed afterwards,
from different caller-supplied parameters, and joined onto the directory it
returned::

    save_path = validate_save_path(save_path, label="save_path")   # checked
    file_path = os.path.join(save_path, f"{filename}.{format}")    # not checked

``os.path.join`` walks back out of its first argument whenever the second asks it
to, so the traversal ``validate_save_path`` refuses in ``save_path`` was reachable
through ``filename`` or ``format`` instead - the halves of the path nothing looked
at. Both are agent-supplied ``@tool`` parameters with no domain of their own, and
the write reported ``status="success"`` at whatever location it had reached.

``resolve_output_path`` closes that by asserting the property that was actually
wanted: the resolved write target lies inside the resolved directory. It is
deliberately not a character allowlist. ``format`` is documented as a closed
vocabulary but #2559 decided it is not held to one, because an unlisted extension
is *honored* rather than replaced - OpenCV writes a ``tiff`` if it can - so
narrowing the spelling would refuse working requests. Containment is orthogonal to
that decision, and these tests pin both halves: a traversal is refused, and every
value that worked before still works, ``tiff`` included.

The sites covered are the ones where a caller-supplied component reaches a path:
``capture`` and ``capture_batch`` (``filename`` and ``format``), ``record``
(``filename``; the ``.mp4`` extension is fixed), and ``PoseManager``
(``robot_id``). ``configure`` is deliberately absent - it composes its name from
``camera_type``, which ``_create_camera`` refuses before the name is built, and a
``camera_id`` whose separators are already stripped.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.lerobot_camera as cam_mod
from strands_robots.tools._path_validation import resolve_output_path, validate_save_path

# Names that resolve outside the directory they are joined onto. The first two are
# the realistic shapes - a traversal in the name, and a traversal reached through
# the extension after a legitimate-looking stem. ``/etc/passwd`` covers the
# absolute name, which ``os.path.join`` honors by discarding its first argument
# entirely.
ESCAPING_NAMES = (
    "../escaped.jpg",
    "../../../../tmp/escaped.jpg",
    "shot.jpg/../../escaped.jpg",
    "/etc/passwd",
    "sub/../../escaped.jpg",
)

# Names that stay inside and must keep working. ``a/b.jpg`` names a subdirectory:
# it is contained, so containment is not what should refuse it.
CONTAINED_NAMES = (
    "shot.jpg",
    "shot.tiff",
    "my capture.jpg",
    "shot..jpg",
    ".hidden.jpg",
    "a/b.jpg",
)


class TestResolveOutputPath:
    """The helper itself."""

    @pytest.mark.parametrize("name", ESCAPING_NAMES)
    def test_a_name_resolving_outside_the_directory_is_refused(self, tmp_path: Path, name: str) -> None:
        root = validate_save_path(str(tmp_path / "captures"))
        os.makedirs(root, exist_ok=True)
        with pytest.raises(ValueError) as excinfo:
            resolve_output_path(root, name, label="filename")
        # The refusal names the value, where it landed and the directory it had to
        # stay in, because a caller cannot correct the name from "invalid" alone.
        message = str(excinfo.value)
        assert "filename" in message
        assert repr(name) in message
        assert root in message

    @pytest.mark.parametrize("name", CONTAINED_NAMES)
    def test_a_contained_name_is_returned_resolved(self, tmp_path: Path, name: str) -> None:
        root = validate_save_path(str(tmp_path / "captures"))
        os.makedirs(root, exist_ok=True)
        resolved = resolve_output_path(root, name)
        assert os.path.isabs(resolved)
        assert os.path.commonpath([root, resolved]) == root

    def test_a_sibling_directory_sharing_a_prefix_is_outside(self, tmp_path: Path) -> None:
        """``startswith`` would accept this; ``commonpath`` must not.

        ``/x/captures2`` shares a string prefix with ``/x/captures`` while being a
        different directory, so a prefix comparison is the wrong relation here.
        """
        root = str(tmp_path / "captures")
        os.makedirs(root, exist_ok=True)
        os.makedirs(str(tmp_path / "captures2"), exist_ok=True)
        with pytest.raises(ValueError, match="outside the directory it must be written into"):
            resolve_output_path(root, "../captures2/shot.jpg")

    def test_a_symlink_cannot_be_used_to_step_outside(self, tmp_path: Path) -> None:
        """Resolution follows symlinks, so a link inside the directory is not a hole."""
        root = tmp_path / "captures"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="outside the directory it must be written into"):
            resolve_output_path(str(root), "link/escaped.jpg")

    def test_naming_the_directory_itself_is_refused_as_such(self, tmp_path: Path) -> None:
        """Reported as naming the directory, not as being outside it.

        This name resolves *to* the directory, so the out-of-bounds wording would
        have to claim it is outside itself.
        """
        root = str(tmp_path / "captures")
        os.makedirs(root, exist_ok=True)
        with pytest.raises(ValueError, match="itself, not a file inside it"):
            resolve_output_path(root, ".")

    @pytest.mark.parametrize("name", ["", "shot\x00.jpg"])
    def test_an_unusable_name_is_refused_before_resolution(self, tmp_path: Path, name: str) -> None:
        root = str(tmp_path / "captures")
        os.makedirs(root, exist_ok=True)
        with pytest.raises(ValueError):
            resolve_output_path(root, name)


class _Recorder:
    """A camera stand-in: connects, yields one frame, records nothing else."""

    def __init__(self) -> None:
        self.width = 8
        self.height = 6
        self.fps = 30
        self.color_mode = type("_M", (), {"value": "RGB"})()

    def connect(self, warmup: bool = True) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def read(self) -> np.ndarray:
        return np.zeros((6, 8, 3), dtype=np.uint8)

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        return self.read()


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Install the stand-in for every camera the tool opens."""
    cam = _Recorder()
    monkeypatch.setattr(cam_mod, "_create_camera", lambda *a, **k: cam)
    return cam


def _text(result: dict[str, Any]) -> str:
    return "\n".join(block["text"] for block in result["content"] if "text" in block)


def _recording_imwrite(written: list[str]) -> Callable[[str, Any], bool]:
    """A ``cv2.imwrite`` stand-in that records its path and reports success.

    Named rather than a lambda because the one-line form would have to be
    ``written.append(path) or True``, and ``list.append`` returns ``None``.
    """

    def imwrite(path: str, img: Any) -> bool:
        written.append(path)
        return True

    return imwrite


class TestTheCameraToolRefusesAnEscapingName:
    """``filename`` and ``format`` are agent-supplied and both reach the path."""

    @pytest.mark.parametrize("action", ["capture", "record"])
    @pytest.mark.parametrize("filename", ["../escaped", "../../../../tmp/escaped"])
    def test_an_escaping_filename_is_refused(
        self, recorder: _Recorder, tmp_path: Path, action: str, filename: str
    ) -> None:
        result = cam_mod.lerobot_camera(
            action=action,
            camera_id=0,
            save_path=str(tmp_path / "captures"),
            filename=filename,
            # A span the rate can honor, so the refusal under test is the name and
            # not the frame count: ``0.01`` at the default ``fps=30`` records no
            # frame and is refused before the path is ever composed.
            capture_duration=0.5,
        )
        assert result["status"] == "error", result
        assert "outside the directory it must be written into" in _text(result)

    def test_an_escaping_format_is_refused(self, recorder: _Recorder, tmp_path: Path) -> None:
        """The extension is the other half of the same composed name."""
        result = cam_mod.lerobot_camera(
            action="capture",
            camera_id=0,
            save_path=str(tmp_path / "captures"),
            filename="shot",
            format="jpg/../../../../tmp/escaped.jpg",
        )
        assert result["status"] == "error", result
        assert "outside the directory it must be written into" in _text(result)

    def test_an_escaping_filename_is_refused_for_a_batch(self, recorder: _Recorder, tmp_path: Path) -> None:
        result = cam_mod.lerobot_camera(
            action="capture_batch",
            camera_ids=[0],
            save_path=str(tmp_path / "captures"),
            filename="../escaped",
        )
        # Batch renders each camera's outcome into the summary text and reports
        # "error" only when no camera succeeded, which is the case here.
        assert result["status"] == "error", result
        assert "outside the directory it must be written into" in _text(result)

    def test_nothing_is_written_outside_the_directory(self, recorder: _Recorder, tmp_path: Path) -> None:
        """The property under test, asserted on the filesystem rather than the message."""
        root = tmp_path / "captures"
        before = sorted(p.name for p in tmp_path.iterdir())
        cam_mod.lerobot_camera(
            action="capture",
            camera_id=0,
            save_path=str(root),
            filename="../escaped",
        )
        # ``captures`` itself is created by the tool before the name is resolved,
        # so it is the only permissible addition.
        after = sorted(p.name for p in tmp_path.iterdir())
        assert [n for n in after if n not in before] == ["captures"]
        assert not (tmp_path / "escaped.jpg").exists()


class TestTheCameraToolStillHonorsEveryWorkingName:
    """Containment must not become a vocabulary, which #2559 decided against."""

    @pytest.mark.parametrize("format", ["jpg", "png", "bmp", "tiff"])
    def test_an_unlisted_extension_is_still_honored(
        self, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, format: str
    ) -> None:
        written: list[str] = []
        monkeypatch.setattr(cam_mod.cv2, "imwrite", _recording_imwrite(written))
        monkeypatch.setattr(os.path, "getsize", lambda path: 1)

        result = cam_mod.lerobot_camera(
            action="capture",
            camera_id=0,
            save_path=str(tmp_path / "captures"),
            filename="shot",
            format=format,
        )

        assert result["status"] == "success", result
        assert written and written[0].endswith(f"shot.{format}")

    def test_a_filename_with_a_space_is_still_honored(
        self, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A containment check has no reason to refuse this; an allowlist would."""
        written: list[str] = []
        monkeypatch.setattr(cam_mod.cv2, "imwrite", _recording_imwrite(written))
        monkeypatch.setattr(os.path, "getsize", lambda path: 1)

        result = cam_mod.lerobot_camera(
            action="capture",
            camera_id=0,
            save_path=str(tmp_path / "captures"),
            filename="my capture",
            format="jpg",
        )

        assert result["status"] == "success", result
        assert written and written[0].endswith("my capture.jpg")


class TestThePoseToolRefusesAnEscapingRobotId:
    """``robot_id`` becomes part of the pose file's name."""

    @pytest.mark.parametrize("robot_id", ["../escaped", "../../../../tmp/escaped"])
    def test_an_escaping_robot_id_is_refused(self, tmp_path: Path, robot_id: str) -> None:
        from strands_robots.tools.pose_tool import PoseManager

        with pytest.raises(ValueError, match="outside the directory it must be written into"):
            PoseManager(robot_id, storage_dir=tmp_path / "poses")

    def test_a_normal_robot_id_still_resolves_inside(self, tmp_path: Path) -> None:
        from strands_robots.tools.pose_tool import PoseManager

        manager = PoseManager("so101_follower", storage_dir=tmp_path / "poses")
        root = os.path.realpath(str(tmp_path / "poses"))
        assert os.path.commonpath([root, str(manager.pose_file)]) == root
        assert manager.pose_file.name == "so101_follower_poses.json"

    def test_the_tool_reports_the_refusal_in_its_own_envelope(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Not as an exception: every other refusal in this tool is an error dict."""
        from strands_robots.tools.pose_tool import pose_tool

        monkeypatch.chdir(tmp_path)
        result = pose_tool(action="list_poses", robot_id="../escaped")

        assert result["status"] == "error", result
        assert "outside the directory it must be written into" in _text(result)
