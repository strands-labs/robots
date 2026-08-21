"""The ``lerobot_camera`` tool must refuse a selector spelling it cannot honor.

Seven numeric options in this agent tool's signature are held to a shared domain
before a camera is opened, and :func:`~strands_robots.tools.lerobot_camera._numeric_option_error`
records why: "every value here is agent-supplied, so each is checked against the
shared domain for its kind before a camera is opened". Two documented enumerated
options sat beside them unchecked, and unlike a frame rate neither is ever refused
downstream - each names a closed vocabulary that ``_create_camera`` resolves by
lookup, and an unrecognised spelling resolved to a *plausible neighbour*:

* ``color_mode`` was ``ColorMode.RGB if color_mode.upper() == "RGB" else ColorMode.BGR``,
  so every value that was not exactly ``RGB`` selected ``BGR``. A trailing space
  (``"rgb "``) or a transposed pair (``"RBG"``) was enough. The driver then delivers
  frames in that channel order while the capture path converts what it is handed
  with ``COLOR_RGB2BGR`` unconditionally, so the saved image has its red and blue
  channels transposed: a red object is written blue, under
  ``status="success"`` and an "Image Capture Success!" summary.
* ``rotation`` was a ``dict.get(rotation.upper(), Cv2Rotation.NO_ROTATION)``, so
  ``"ROTATE_90_DEG"`` or ``"90"`` silently selected no rotation at all - the caller
  receives a correct image of the wrong orientation, again reported as success.

Neither substitution appears anywhere in the tool result, which is what separates
this from a value the device would have rejected: the request is honored, just not
the one that was made.

These tests pin the domain, the per-action scoping (an option an action never reads
must not be refused), the case-insensitivity the previous behaviour already had, and
that ``_create_camera`` resolves through the same vocabularies the gate validates
against, so the accepted set and the enforced set cannot drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# The module object itself is needed - for the ``cv2`` handle the tool module
# imports and for the vocabulary maps the drift guards read - so every name in it,
# the tool included, is reached through this one alias.
import strands_robots.tools.lerobot_camera as cam_mod

# Actions that open a camera configured with the caller's selectors. Every one of
# them must therefore have those selectors validated.
CAMERA_ACTIONS = ("capture", "capture_batch", "record", "preview", "test", "configure")

# One value per rejection reason. ``"rgb "`` and ``"RBG"`` are the realistic slips
# (a trailing space, a transposed pair); the rest cover a wrong vocabulary, an
# empty string and the non-string shapes an agent can put on the wire.
BAD_COLOR_MODES = ("rgb ", "RBG", "GBR", "grayscale", "", None, 0, ["RGB"])
BAD_ROTATIONS = ("ROTATE_90_DEG", "90", "rotate 90", "ROTATE_45", "", None, 90, ["ROTATE_90"])


class _Recorder:
    """A camera stand-in that records the selectors it was configured with."""

    def __init__(self) -> None:
        self.opened: list[tuple[Any, Any]] = []
        self.width = 8
        self.height = 6
        self.fps = 30
        self.color_mode = type("_M", (), {"value": "RGB"})()
        self.rotation: Any = None

    def connect(self, warmup: bool = True) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def read(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Substitute the camera factory and record every selector pair it is handed."""
    cam = _Recorder()

    def _create(
        camera_type: str,
        camera_id: Any,
        width: Any,
        height: Any,
        fps: Any,
        color_mode: Any,
        rotation: Any,
    ) -> _Recorder:
        cam.opened.append((color_mode, rotation))
        return cam

    monkeypatch.setattr(cam_mod, "_create_camera", _create)
    return cam


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool with agent-shaped keyword values.

    The tool annotates both selectors ``str``, but an agent supplies JSON, and the
    values under test here are deliberately outside that annotation - a ``None``
    mode, an integer rotation. Funnelling every call through one ``**kwargs: Any``
    helper states that once rather than scattering a suppression over each call.
    """
    return cam_mod.lerobot_camera(**kwargs)


class TestASelectorThatSilentlyChoseAnotherIsRefused:
    """The headline: an unrecognised spelling used to pick a plausible neighbour."""

    @pytest.mark.parametrize("action", CAMERA_ACTIONS)
    @pytest.mark.parametrize("value", BAD_COLOR_MODES)
    def test_an_unrecognised_color_mode_is_refused_on_every_camera_action(
        self, recorder: _Recorder, tmp_path: Any, action: str, value: Any
    ) -> None:
        result = _call(action=action, camera_id=0, save_path=str(tmp_path), color_mode=value)

        assert result["status"] == "error", result
        assert "color_mode" in _text(result)
        # The refusal precedes the device, so no camera is opened with a channel
        # order the caller did not ask for.
        assert recorder.opened == []

    @pytest.mark.parametrize("action", CAMERA_ACTIONS)
    @pytest.mark.parametrize("value", BAD_ROTATIONS)
    def test_an_unrecognised_rotation_is_refused_on_every_camera_action(
        self, recorder: _Recorder, tmp_path: Any, action: str, value: Any
    ) -> None:
        result = _call(action=action, camera_id=0, save_path=str(tmp_path), rotation=value)

        assert result["status"] == "error", result
        assert "rotation" in _text(result)
        assert recorder.opened == []

    def test_the_refusal_names_the_action_the_option_and_the_vocabulary(
        self, recorder: _Recorder, tmp_path: Any
    ) -> None:
        text = _text(_call(action="capture", camera_id=0, save_path=str(tmp_path), color_mode="RBG"))

        assert text.startswith("capture:")
        assert "color_mode" in text
        # The accepted set is in the message, so a caller does not have to go and
        # read the docstring to recover.
        assert "RGB" in text and "BGR" in text
        assert "'RBG'" in text
        assert all(ord(c) < 128 for c in text), text


class TestTheChannelOrderThatWasSilentlyTransposed:
    """The harm behind the ``color_mode`` half, measured on the saved pixels."""

    @pytest.fixture
    def honoring_camera(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Substitute a camera that delivers the channel order it is configured with.

        This is what makes the transposition observable: the real driver honors the
        ``ColorMode`` in its config, and the capture path then applies
        ``COLOR_RGB2BGR`` to whatever it is handed regardless.
        """

        class _Honoring:
            def __init__(self, bgr: bool) -> None:
                self.bgr = bgr
                self.color_mode = type("_M", (), {"value": "BGR" if bgr else "RGB"})()
                self.rotation = None

            def connect(self, warmup: bool = True) -> None:
                return None

            def disconnect(self) -> None:
                return None

            def read(self) -> np.ndarray:
                frame: np.ndarray = np.zeros((4, 4, 3), dtype=np.uint8)
                frame[:, :, 0] = 255  # a red scene, in RGB order
                if self.bgr:
                    frame = frame[:, :, ::-1].copy()
                return frame

            def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
                return self.read()

        def _create(
            camera_type: str,
            camera_id: Any,
            width: Any,
            height: Any,
            fps: Any,
            color_mode: Any,
            rotation: Any,
        ) -> _Honoring:
            return _Honoring(bgr=str(color_mode).upper() != "RGB")

        monkeypatch.setattr(cam_mod, "_create_camera", _create)

    def test_a_declared_color_mode_writes_the_scene_it_photographed(self, honoring_camera: None, tmp_path: Any) -> None:
        result = _call(
            action="capture", camera_id=0, save_path=str(tmp_path), filename="shot", color_mode="RGB", format="png"
        )

        assert result["status"] == "success", result
        # cv2.imread yields BGR order, so a red scene has its last channel hot.
        pixel = cam_mod.cv2.imread(str(tmp_path / "shot.png"))
        assert pixel is not None, "the capture wrote no decodable image"
        blue, _green, red = (int(pixel[0, 0, 0]), int(pixel[0, 0, 1]), int(pixel[0, 0, 2]))
        assert red > blue, f"expected a red pixel, got BGR {(blue, _green, red)}"

    def test_a_trailing_space_no_longer_writes_that_scene_transposed(
        self, honoring_camera: None, tmp_path: Any
    ) -> None:
        result = _call(
            action="capture", camera_id=0, save_path=str(tmp_path), filename="shot", color_mode="rgb ", format="png"
        )

        assert result["status"] == "error", result
        # Nothing is written, so there is no transposed image to hand back and no
        # "Image Capture Success!" summary naming one.
        assert list(tmp_path.iterdir()) == []


class TestEveryDeclaredSpellingStillWorks:
    """The fix refuses only the spellings that used to be silently replaced."""

    @pytest.mark.parametrize("value", ["RGB", "BGR", "rgb", "bgr", "Rgb"])
    def test_a_declared_color_mode_is_accepted_in_any_case(
        self, recorder: _Recorder, tmp_path: Any, value: str
    ) -> None:
        result = _call(action="capture", camera_id=0, save_path=str(tmp_path), color_mode=value)

        assert result["status"] == "success", result
        assert recorder.opened == [(value, "NO_ROTATION")]

    @pytest.mark.parametrize("value", ["NO_ROTATION", "ROTATE_90", "ROTATE_180", "ROTATE_270", "rotate_90"])
    def test_a_declared_rotation_is_accepted_in_any_case(self, recorder: _Recorder, tmp_path: Any, value: str) -> None:
        result = _call(action="capture", camera_id=0, save_path=str(tmp_path), rotation=value)

        assert result["status"] == "success", result
        assert recorder.opened == [("RGB", value)]

    def test_the_default_selectors_are_themselves_declared(self, recorder: _Recorder, tmp_path: Any) -> None:
        """A caller who names neither selector is never refused for the defaults."""
        result = _call(action="capture", camera_id=0, save_path=str(tmp_path))

        assert result["status"] == "success", result


class TestOnlyTheOptionsAnActionReadsAreValidated:
    """An action that applies neither selector must not be refused for one."""

    @pytest.mark.parametrize("action", ["discover", "list"])
    def test_an_action_that_opens_no_configured_camera_ignores_the_selectors(self, action: str) -> None:
        text = _text(_call(action=action, color_mode="nonsense", rotation="nonsense"))

        # These probe devices without applying either selector, so the value is
        # genuinely inert and refusing it would be a false rejection. Matched on
        # the refusal's own phrase rather than the bare option names: "list"
        # legitimately reports "Supported rotations: 0, 90, 180, 270 degrees".
        assert "must be one of" not in text, text
        assert not text.startswith(f"{action}:"), text

    def test_the_image_format_is_left_to_its_own_consumers(self, recorder: _Recorder, tmp_path: Any) -> None:
        """``format`` is deliberately outside this domain.

        Unlike the two selectors, an unlisted ``format`` is *honored* rather than
        replaced - it becomes the saved file's extension and OpenCV writes it if it
        can - so narrowing it to the three names the docstring lists would refuse
        working requests. Its failure mode is loud, not silent.
        """
        result = _call(action="capture", camera_id=0, save_path=str(tmp_path), format="tiff")

        assert result["status"] == "success", result
        # Matched on the refusal's own phrase: pytest derives ``tmp_path`` from the
        # test name, so a bare "format" appears in the reported save path.
        assert "format must be one of" not in _text(result)


class TestTheVocabulariesCannotDriftFromTheResolver:
    """The gate and ``_create_camera`` must read the same vocabularies."""

    def test_every_camera_opening_action_has_a_vocabulary_row(self) -> None:
        # The two tables answer the same question - which actions open a camera
        # with caller-supplied configuration - so they must name the same actions.
        assert set(cam_mod._ACTION_VOCABULARY_OPTIONS) == set(cam_mod._ACTION_NUMERIC_OPTIONS)
        assert set(cam_mod._ACTION_VOCABULARY_OPTIONS) == set(CAMERA_ACTIONS)
        for action, options in cam_mod._ACTION_VOCABULARY_OPTIONS.items():
            assert set(options) == {"color_mode", "rotation"}, action

    @pytest.mark.parametrize("spelling", ["RGB", "BGR"])
    def test_every_accepted_color_mode_resolves_to_a_distinct_enum(
        self, monkeypatch: pytest.MonkeyPatch, spelling: str
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_config(index_or_path: Any, fps: Any, width: Any, height: Any, color_mode: Any, rotation: Any) -> Any:
            seen["color_mode"] = color_mode
            return object()

        monkeypatch.setattr(cam_mod, "OpenCVCameraConfig", fake_config)
        monkeypatch.setattr(cam_mod, "OpenCVCamera", lambda config: config)

        cam_mod._create_camera("opencv", 0, 8, 6, 30, spelling, "NO_ROTATION")

        assert seen["color_mode"] is cam_mod._COLOR_MODES[spelling]

    @pytest.mark.parametrize("spelling", ["NO_ROTATION", "ROTATE_90", "ROTATE_180", "ROTATE_270"])
    def test_every_accepted_rotation_resolves_to_a_distinct_enum(
        self, monkeypatch: pytest.MonkeyPatch, spelling: str
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_config(index_or_path: Any, fps: Any, width: Any, height: Any, color_mode: Any, rotation: Any) -> Any:
            seen["rotation"] = rotation
            return object()

        monkeypatch.setattr(cam_mod, "OpenCVCameraConfig", fake_config)
        monkeypatch.setattr(cam_mod, "OpenCVCamera", lambda config: config)

        cam_mod._create_camera("opencv", 0, 8, 6, 30, "RGB", spelling)

        assert seen["rotation"] is cam_mod._ROTATIONS[spelling]

    def test_the_documented_vocabularies_are_the_enforced_ones(self) -> None:
        """The public docstring must name exactly the accepted spellings."""
        doc = cam_mod.lerobot_camera.__doc__ or ""
        for spelling in cam_mod._COLOR_MODES:
            assert f'"{spelling}"' in doc, spelling
        for spelling in cam_mod._ROTATIONS:
            assert f'"{spelling}"' in doc, spelling

    def test_the_numeric_domain_beside_it_is_unchanged(self, recorder: _Recorder, tmp_path: Any) -> None:
        """The sibling gate still refuses a rate no camera can honor."""
        result = _call(action="capture", camera_id=0, save_path=str(tmp_path), fps=0)

        assert result["status"] == "error"
        assert "fps" in _text(result)
