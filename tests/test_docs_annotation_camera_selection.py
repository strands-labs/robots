"""Keep the annotation guide honest about which camera the labels come from.

``lerobot-annotate``'s ``plan`` and ``interjections`` modules read exactly one
video stream - the dataset's first video key. ``VideoFrameProvider`` resolves
that default in ``lerobot/annotations/steerable_pipeline/frames.py``
(``self.camera_key = keys[0]``) and both modules call ``frames_at()`` without
naming a camera, so a multi-camera dataset gets every subtask/plan/memory label
derived from one view.

Which view that is comes straight out of ``start_recording``: the recorder
writes its video features in the caller's ``cameras=`` order, so the first name
in that list becomes the first video key. That makes the choice load-bearing,
and it is a bad choice when the first camera is gripper-mounted: a camera added
with ``parent_body="<arm>/gripper"`` travels with whatever the gripper holds, so
a carried object is pinned in its frame while a stationary object sweeps out of
it. The image evidence for "the object moved" is then present exactly when the
object did not move.

Two assertions, deliberately of different kinds:

* the ordering contract the guidance rests on is measured against a real
  recording (a caller that lists a world-fixed camera first really does get it
  as the first video key), and
* the guide names the escape hatch - the ``camera_key`` field ``lerobot``'s
  ``VlmConfig`` actually exposes, read out of that dataclass rather than
  hard-coded here - and warns about the gripper-mounted case.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOC = _REPO_ROOT / "docs" / "data" / "annotation.md"


def _doc_text() -> str:
    return _DOC.read_text(encoding="utf-8")


def _vlm_config_camera_field() -> str:
    """The ``VlmConfig`` field that selects the stream, read from lerobot's source.

    Parsed rather than imported: importing the annotation package pulls the whole
    pipeline (and its optional deps) for one dataclass field name.
    """
    __tracebackhide__ = True
    lerobot = pytest.importorskip("lerobot")
    config_py = Path(lerobot.__file__).parent / "annotations" / "steerable_pipeline" / "config.py"
    if not config_py.is_file():
        raise pytest.skip.Exception(f"lerobot annotation pipeline not present at {config_py}")
    tree = ast.parse(config_py.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VlmConfig":
            fields = [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
            camera_fields = [name for name in fields if "camera" in name]
            assert camera_fields, (
                "lerobot's VlmConfig no longer exposes a camera-selection field; the "
                "annotation guide's advice needs rewriting, not just renaming."
            )
            return camera_fields[0]
    raise pytest.skip.Exception("lerobot's VlmConfig not found in the installed annotation pipeline")


class TestTheGuideNamesTheStreamSelector:
    """The guide must say which view the labels come from, and how to change it."""

    def test_guide_names_the_vlm_camera_field(self) -> None:
        field = _vlm_config_camera_field()
        text = _doc_text()
        assert f"--vlm.{field}" in text, (
            f"docs/data/annotation.md never mentions --vlm.{field}, so a reader "
            "cannot tell that the plan/interjections modules read only the first "
            "video key, nor how to point them somewhere else."
        )

    def test_guide_warns_about_a_gripper_mounted_first_camera(self) -> None:
        text = _doc_text().lower()
        assert "parent_body" in text, (
            "docs/data/annotation.md does not mention parent_body, so it never "
            "warns that a gripper-mounted camera is the wrong stream to derive "
            "scene-level motion labels from."
        )

    def test_guide_explains_the_recording_order_lever(self) -> None:
        text = _doc_text()
        assert "cameras=" in text, (
            "docs/data/annotation.md does not mention the cameras= recording "
            "argument, which is what decides the first video key."
        )


class TestTheRecordedOrderFollowsTheCaller:
    """The lever the guide points at has to be real: the recorded video-feature
    order is the caller's ``cameras=`` order, so listing a world-fixed camera
    first really does make it the stream the shared modules read."""

    def test_first_video_key_is_the_callers_first_camera(self, tmp_path: Path) -> None:
        pytest.importorskip("mujoco")
        pytest.importorskip("lerobot")
        from strands_robots.simulation.mujoco.backend import _can_render

        if not _can_render():
            pytest.skip("No GL context available (headless CI without EGL/OSMesa)")
        from strands_robots.dataset_recorder import has_lerobot_dataset
        from strands_robots.simulation.mujoco.simulation import Simulation

        if not has_lerobot_dataset():
            pytest.skip("dataset recording needs the [lerobot] extra")

        sim = Simulation(tool_name="annotation_camera_order", mesh=False)
        try:
            sim.create_world()
            added = sim.add_robot(name="arm", data_config="so101")
            assert added["status"] == "success", added
            for name, kwargs in (
                ("wrist", {"parent_body": "arm/gripper", "position": [0.0, 0.1, 0.0]}),
                ("front", {"position": [-0.5, -0.1, 0.25], "target": [-0.2, -0.1, 0.07]}),
            ):
                cam = sim.add_camera(name=name, width=64, height=64, **kwargs)
                assert cam["status"] == "success", (name, cam)

            # A caller who wants the world-fixed view to drive the labels lists it
            # first; the recorded schema has to honour that order.
            root = tmp_path / "ds"
            started = sim.start_recording(
                repo_id="local/annotation_camera_order",
                task="probe",
                fps=30,
                root=str(root),
                overwrite=True,
                cameras=["front", "wrist"],
            )
            assert started["status"] == "success", started
            ran = sim.run_policy(
                robot_name="arm",
                policy_provider="mock",
                n_steps=2,
                control_frequency=30.0,
                fast_mode=True,
            )
            assert ran["status"] == "success", ran
            stopped = sim.stop_recording()
            assert stopped["status"] == "success", stopped
        finally:
            sim.cleanup()

        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
        assert video_keys[:2] == [
            "observation.images.front",
            "observation.images.wrist",
        ], (
            "the recorded video-feature order no longer follows the caller's "
            f"cameras= order (got {video_keys}), so listing a world-fixed camera "
            "first is no longer enough to keep it out of the annotation "
            "pipeline's default stream and the guide's advice is stale."
        )
