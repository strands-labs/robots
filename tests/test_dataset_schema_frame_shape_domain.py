"""``DatasetRecorder.create`` refuses a camera frame shape it cannot honor.

``camera_dims`` and the ``video_width`` / ``video_height`` pair are one quantity
in two spellings. ``_build_features`` reads
``camera_dims.get(camera, (video_height, video_width))`` once per declared
camera, so the mapping sets the shape of the cameras it covers and the pair sets
the shape of every other one. The shape is a *declaration*, not a resize - the
recorder rescales nothing - so whatever is given goes straight into the LeRobot
feature as ``(3, height, width)`` and is not compared against a real frame until
the first ``add_frame``.

Three mistakes could not be honored as written, and none of them was reported
anywhere near the parameter that caused it:

* **A key ``camera_keys`` does not declare is dropped by that ``.get``.** This
  is the quiet one. Nothing is logged, the dataset is created, and the camera
  the entry was meant for silently takes the global pair instead - so a camera
  streaming 240x320, declared as ``camera_dims={"imagee": (240, 320)}`` against
  ``camera_keys=["image"]``, was declared ``(3, 480, 640)`` from the defaults.
* **A component that is not a positive integer is written in as given**, so the
  schema declared ``(3, 480, nan)``, ``(3, 480, '640')``, ``(3, 480, [640])``
  or ``(3, 480, True)`` and no frame could ever match it.
* **A value that is not a two-element sequence** unpacks as a bare ``TypeError``
  / ``ValueError`` (``cannot unpack non-iterable int object``), and a
  non-mapping ``camera_dims`` as a bare ``AttributeError`` from the ``.get``
  (``'list' object has no attribute 'get'``) - none of which names this
  parameter or this method.

The component domain is the shared
:func:`~strands_robots.utils.positive_count_error`, which is the strict-``int``
one because a pixel count here is written into ``meta/info.json``: an integral
float is declared as ``480.0`` and a ``numpy`` integer raises
``TypeError: Object of type int64 is not JSON serializable`` from the metadata
write, so the wider whole-number domain would accept values this consumer
refuses.

The refusal sits in the same guard block, and is placed ahead of the same two
side effects, as the schema *column name* lists checked by the sibling module
``tests/test_dataset_schema_column_names_distinct.py``: ahead of the lazy
lerobot import, so the same caller mistake reports identically on a minimal
install (which is why every refusal test here runs without the extra), and ahead
of the on-disk target, which ``overwrite=True`` removes.
"""

import ast
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots import dataset_recorder as recorder_mod
from strands_robots.utils import positive_count_error

# Reached through the module alias rather than a second from-import: the
# guard-placement tests read the module's own source, and one handle for one
# module keeps that unambiguous.
DatasetRecorder = recorder_mod.DatasetRecorder

# Values no pixel count can be built from. ``positive_count_error`` owns this
# domain; the labels are reused as parametrize ids.
UNUSABLE_COUNTS: list[tuple[str, Any]] = [
    ("zero", 0),
    ("negative", -64),
    ("fractional", 2.7),
    ("integral_float", 320.0),
    ("bool", True),
    ("nan", math.nan),
    ("inf", math.inf),
    ("numeric_string", "640"),
    ("none", None),
    ("list", [640]),
    ("numpy_int", np.int64(320)),
]

# Pairs that are not a ``(height, width)`` of two components.
UNUSABLE_PAIRS: list[tuple[str, Any]] = [
    ("scalar", 480),
    ("one_element", (480,)),
    ("three_elements", (480, 640, 3)),
    ("none", None),
    ("string_of_two", "hw"),
    ("mapping", {"height": 480, "width": 640}),
    ("zero_dim_array", np.array(480)),
]

# ``camera_dims`` values that are not a mapping at all.
NON_MAPPINGS: list[tuple[str, Any]] = [
    ("list_of_pairs", [(240, 320)]),
    ("bare_string", "image"),
    ("scalar", 480),
    ("tuple", (240, 320)),
]


class _FakeLeRobotDataset:
    """Stand-in for ``LeRobotDataset``, recording whether ``create`` was reached."""

    calls: list[dict[str, Any]] = []

    def __init__(self, features: dict[str, Any]) -> None:
        self.repo_id = "local/fake"
        self.features = features
        self.meta = None

    @classmethod
    def create(cls, **kwargs: Any) -> "_FakeLeRobotDataset":
        cls.calls.append(kwargs)
        return cls(kwargs.get("features", {}))


def _create(**kwargs: Any) -> DatasetRecorder:
    """Call ``DatasetRecorder.create`` with keywords the signature disallows.

    Most tests here pass a value the parameter's annotation forbids (a bare
    string where ``int`` is declared, a scalar where a ``(height, width)`` pair
    is) because that is precisely the caller mistake the runtime guard exists
    for. Routing them through one ``**kwargs: Any`` funnel states that intent
    once instead of scattering per-call type suppressions.
    """
    return DatasetRecorder.create(**kwargs)


@pytest.fixture
def fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLeRobotDataset]:
    """Route ``create`` onto a fake dataset class, so no lerobot extra is needed."""
    _FakeLeRobotDataset.calls = []
    monkeypatch.setattr(recorder_mod, "_get_lerobot_dataset_class", lambda: _FakeLeRobotDataset)
    return _FakeLeRobotDataset


@pytest.fixture
def no_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make reaching the lazy lerobot import fatal.

    A refusal must be decided before ``create`` probes the dataset extra, so the
    same caller mistake is reported the same way on a minimal install.
    """

    def _fatal() -> Any:
        raise AssertionError("the lerobot extra was probed before the frame shape was checked")

    monkeypatch.setattr(recorder_mod, "_get_lerobot_dataset_class", _fatal)


class TestAnEntryForAnUndeclaredCameraIsRefused:
    """The quiet half: an unlooked-up entry, and the camera declared at the pair."""

    def test_a_mistyped_camera_name_is_refused_and_named(self, no_lerobot: None) -> None:
        with pytest.raises(ValueError) as excinfo:
            _create(
                repo_id="local/probe",
                camera_keys=["image"],
                camera_dims={"imagee": (240, 320)},
            )
        text = str(excinfo.value)
        assert "camera_dims" in text, text
        assert "'imagee'" in text, text
        assert "Did you mean 'image'?" in text, text
        assert "['image']" in text, text

    def test_an_unrelated_key_is_refused_without_inventing_a_suggestion(self, no_lerobot: None) -> None:
        """A near match is offered; an unrelated name only gets the declared list."""
        with pytest.raises(ValueError) as excinfo:
            _create(
                repo_id="local/probe",
                camera_keys=["image", "wrist_image"],
                camera_dims={"joint1": (240, 320)},
            )
        text = str(excinfo.value)
        assert "Did you mean" not in text, text
        assert "['image', 'wrist_image']" in text, text

    def test_dims_for_a_camera_less_dataset_are_refused(self, no_lerobot: None) -> None:
        """With no ``camera_keys`` no image feature is built, so every entry is ignored."""
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", joint_names=["j1"], camera_dims={"image": (240, 320)})
        text = str(excinfo.value)
        assert "no camera is declared" in text, text
        assert "camera_keys" in text, text

    def test_the_declared_camera_keeps_its_own_shape(self, fake_lerobot: type[_FakeLeRobotDataset]) -> None:
        """The control: spelled correctly, the entry wins over the global pair."""
        _create(
            repo_id="local/probe",
            camera_keys=["image"],
            camera_dims={"image": (240, 320)},
            video_width=640,
            video_height=480,
        )
        features = fake_lerobot.calls[0]["features"]
        assert features["observation.images.image"]["shape"] == (3, 240, 320)


class TestUnusableFrameShapesAreRefused:
    """Neither spelling of the shape accepts a component that is not a pixel count."""

    @pytest.mark.parametrize("param", ["video_width", "video_height"])
    @pytest.mark.parametrize(("label", "value"), UNUSABLE_COUNTS, ids=[c[0] for c in UNUSABLE_COUNTS])
    def test_global_pair_component_refused(self, no_lerobot: None, param: str, label: str, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", camera_keys=["image"], **{param: value})
        text = str(excinfo.value)
        assert param in text, text
        assert "must be a positive integer" in text, text
        assert "DatasetRecorder.create" in text, text

    @pytest.mark.parametrize("axis", ["height", "width"])
    @pytest.mark.parametrize(("label", "value"), UNUSABLE_COUNTS, ids=[c[0] for c in UNUSABLE_COUNTS])
    def test_per_camera_component_refused(self, no_lerobot: None, axis: str, label: str, value: Any) -> None:
        dims = (value, 320) if axis == "height" else (240, value)
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", camera_keys=["image"], camera_dims={"image": dims})
        text = str(excinfo.value)
        assert f"camera_dims['image'] {axis}" in text, text
        assert "must be a positive integer" in text, text

    @pytest.mark.parametrize(("label", "value"), UNUSABLE_PAIRS, ids=[c[0] for c in UNUSABLE_PAIRS])
    def test_a_value_that_is_not_a_height_width_pair_is_refused(self, no_lerobot: None, label: str, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", camera_keys=["image"], camera_dims={"image": value})
        text = str(excinfo.value)
        assert "camera_dims['image']" in text, text
        assert "(height, width) pair" in text, text

    @pytest.mark.parametrize(("label", "value"), NON_MAPPINGS, ids=[c[0] for c in NON_MAPPINGS])
    def test_a_non_mapping_camera_dims_is_refused(self, no_lerobot: None, label: str, value: Any) -> None:
        """Its entries are looked up per camera, so a non-mapping cannot be read."""
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", camera_keys=["image"], camera_dims=value)
        text = str(excinfo.value)
        assert "camera_dims must be a mapping" in text, text
        assert type(value).__name__ in text, text


class TestUsableFrameShapesStillReachTheDataset:
    """The over-reach controls: nothing that could be honored is refused."""

    @pytest.mark.parametrize(
        ("label", "kwargs"),
        [
            ("defaults", {}),
            ("explicit_pair", {"video_width": 320, "video_height": 240}),
            ("dims_none", {"camera_dims": None}),
            ("dims_empty", {"camera_dims": {}}),
            ("dims_tuple", {"camera_dims": {"image": (240, 320)}}),
            ("dims_list", {"camera_dims": {"image": [240, 320]}}),
            ("dims_and_pair", {"camera_dims": {"image": (240, 320)}, "video_width": 640}),
        ],
    )
    def test_accepted(self, fake_lerobot: type[_FakeLeRobotDataset], label: str, kwargs: dict[str, Any]) -> None:
        _create(repo_id="local/probe", camera_keys=["image"], **kwargs)
        assert len(fake_lerobot.calls) == 1

    def test_a_list_pair_is_honored_as_given(self, fake_lerobot: type[_FakeLeRobotDataset]) -> None:
        """A list is accepted because the consumer honors it - the annotation says tuple."""
        _create(repo_id="local/probe", camera_keys=["image"], camera_dims={"image": [240, 320]})
        features = fake_lerobot.calls[0]["features"]
        assert features["observation.images.image"]["shape"] == (3, 240, 320)


class TestARefusedCreateLeavesTheTargetAlone:
    """The refusal precedes the on-disk work, so nothing is destroyed by it."""

    def test_overwrite_does_not_remove_an_existing_dataset(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset]
    ) -> None:
        """``overwrite=True`` removes the target directory - a refusal must not."""
        root = tmp_path / "ds"
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text('{"fps": 30}')

        raised: Exception | None = None
        try:
            _create(
                repo_id="local/probe",
                root=str(root),
                camera_keys=["image"],
                camera_dims={"imagee": (240, 320)},
                overwrite=True,
            )
        except ValueError as exc:
            raised = exc

        # Assert the surviving target first: that is the consequence that matters,
        # and it is what a refusal arriving after ``_prepare_create_target`` would
        # already have destroyed.
        assert (root / "meta" / "info.json").is_file(), "the refused call deleted the dataset"
        assert fake_lerobot.calls == [], "the refused call still built a dataset"
        assert raised is not None, "the undeclared camera_dims key was not refused"
        assert "is not a declared camera" in str(raised), str(raised)

    def test_a_usable_call_does_reach_the_dataset_target(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset]
    ) -> None:
        """The control: the same call with the key spelled correctly is not refused."""
        _create(
            repo_id="local/probe",
            root=str(tmp_path / "fresh"),
            camera_keys=["image"],
            camera_dims={"image": (240, 320)},
        )
        assert len(fake_lerobot.calls) == 1


class TestTheDomainCannotDriftFromTheSharedRule:
    """The component check delegates rather than restating the numeric domain."""

    @pytest.mark.parametrize(
        ("label", "value"),
        UNUSABLE_COUNTS + [("plain_int", 320), ("large", 4096)],
        ids=[c[0] for c in UNUSABLE_COUNTS] + ["plain_int", "large"],
    )
    def test_both_spellings_agree_with_positive_count_error(self, label: str, value: Any) -> None:
        """A component refused as ``video_width`` is refused inside ``camera_dims`` too."""
        shared_refuses = positive_count_error(value, "x", "y") is not None
        via_pair = recorder_mod._frame_shape_error(None, ["image"], value, 480) is not None
        via_dims = recorder_mod._frame_shape_error({"image": (480, value)}, ["image"], 640, 480) is not None
        assert via_pair is shared_refuses, f"video_width verdict differs for {value!r}"
        assert via_dims is shared_refuses, f"camera_dims verdict differs for {value!r}"

    def test_the_message_is_the_shared_one_verbatim(self) -> None:
        """No second wording for the same rule."""
        assert recorder_mod._frame_shape_error(None, ["image"], 0, 480) == positive_count_error(
            0, "video_width", "DatasetRecorder.create"
        )

    def test_the_helper_reports_no_problem_for_a_usable_shape(self) -> None:
        assert recorder_mod._frame_shape_error({"image": (240, 320)}, ["image"], 640, 480) is None
        assert recorder_mod._frame_shape_error(None, [], 0, 0) is None


class TestGuardPlacement:
    """``create`` checks the shape before either of its two side effects."""

    def test_the_call_forwards_every_parameter_of_the_quantity(self) -> None:
        """A future edit cannot drop one of the four from the guard call."""
        source = inspect.getsource(DatasetRecorder.create)
        tree = ast.parse(inspect.cleandoc(source).replace("@classmethod\n", "", 1))
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)

        call = next(
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_frame_shape_error"
        )
        forwarded = [ast.unparse(arg) for arg in call.args]
        assert forwarded == ["camera_dims", "camera_keys", "video_width", "video_height"], forwarded

    def test_the_shape_is_checked_before_the_lerobot_probe_and_the_target(self) -> None:
        """Source order: name lists, then the shape, then any work."""
        source = inspect.getsource(DatasetRecorder.create)
        tree = ast.parse(inspect.cleandoc(source).replace("@classmethod\n", "", 1))
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)

        def called_names(node: ast.AST) -> set[str]:
            return {
                child.func.attr if isinstance(child.func, ast.Attribute) else getattr(child.func, "id", "")
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
            }

        index_of = {
            name: i
            for i, stmt in enumerate(func.body)
            for name in called_names(stmt)
            if name in {"name_list_error", "_frame_shape_error", "_get_lerobot_dataset_class"}
        }
        assert index_of["name_list_error"] < index_of["_frame_shape_error"], index_of
        assert index_of["_frame_shape_error"] < index_of["_get_lerobot_dataset_class"], index_of


class TestNeighbouringSurfacesStayOutOfScope:
    """Premise pins, so a follow-up is not closed by accident.

    ``fps`` was the recorder's other unvalidated schema option when this module
    landed, and these pins asserted that it still reached the dataset unchecked -
    so that #2068 could not be closed by accident. It has since been refused on
    its own shared domain (a rate, not a frame shape:
    :func:`~strands_robots.utils.positive_whole_number_error`), so the pins are
    REPLACED by the opposite statement rather than deleted. What they are here to
    say either way is that the two quantities are guarded independently: neither
    refusal can be what makes the other's test pass.
    """

    @pytest.mark.parametrize("value", [2.7, math.nan, math.inf, True])
    def test_an_unusable_fps_is_refused_without_naming_the_frame_shape(self, no_lerobot: None, value: Any) -> None:
        """Refused on the rate domain - see tests/test_dataset_recorder_fps_domain.py."""
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", camera_keys=["image"], fps=value)
        text = str(excinfo.value)
        assert "fps must be a positive whole number" in text, text
        assert "camera_dims" not in text, text
        assert "video_width" not in text, text

    def test_a_usable_fps_leaves_the_shape_refusals_intact(self, no_lerobot: None) -> None:
        """The control: a usable rate does not shadow the frame-shape refusal."""
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", camera_keys=["image"], video_width=0, fps=30)
        assert "video_width" in str(excinfo.value)

    def test_a_camera_less_create_does_not_read_the_pair(self, fake_lerobot: type[_FakeLeRobotDataset]) -> None:
        """With no camera declared the pair decides nothing, so it is left alone."""
        _create(repo_id="local/probe", joint_names=["j1"], video_width=0, video_height=-8)
        assert len(fake_lerobot.calls) == 1
