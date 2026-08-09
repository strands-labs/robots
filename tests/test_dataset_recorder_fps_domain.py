"""``DatasetRecorder.create`` refuses a recording rate it cannot honor.

``fps`` is the rate the dataset is DECLARED at: it is written into
``meta/info.json`` and every frame timestamp is derived from it. Nothing is
throttled to it, so the value is never compared against a real capture rate -
whatever is given is what the dataset claims.

LeRobot rejects only ``fps <= 0``, so everything else was ours to refuse and was
refused nowhere. Measured on ``6de04db3`` (one ``create`` + one ``add_frame`` +
``save_episode`` + ``finalize`` per row):

===========  ==============  ====================  ============
``fps``      ``create()``    ``meta.fps`` on disk  frames saved
===========  ==============  ====================  ============
``30``       success         ``30``                1
``0``/``-5`` ValueError      -                     - (LeRobot's own check)
``2.7``      success         ``2.7``               **0**
``nan``      success         ``nan``               **0**
``inf``      success         ``inf``               **0**
``True``     success         ``True``              1, at 1 fps
``'30'``     bare TypeError  -                     -
===========  ==============  ====================  ============

The three quiet rows are the ones that cost an episode: ``create``,
``add_frame``, ``save_episode`` and ``finalize`` all returned normally and the
episode held zero frames. ``True`` is quiet in the other direction - an ``int``
subclass, so it recorded a 1 fps dataset with no complaint. ``'30'`` and ``None``
leaked a bare ``TypeError: '<=' not supported between instances of 'str' and
'int'`` naming neither the parameter nor the method, out of a constructor
documented to raise ``ValueError`` / ``FileExistsError``.

**The domain is the facades' domain, by the same function.** Every backend's
``start_recording`` already applies
:func:`~strands_robots.simulation.recording.dataset_recording_option_error` - a
``{"status": "error"}`` envelope around
:func:`~strands_robots.utils.positive_whole_number_error` - and then forwards
``fps`` here unchanged. So the facades were covered and only the documented
direct API was not, and the rule at this depth must be *the same one*: a narrower
domain here would refuse a value ``start_recording`` had already reported usable,
turning an error envelope into a ``ValueError`` raised out of a method whose
contract is to return one. ``TestTheRuleIsTheFacadesRule`` pins that agreement
value by value rather than trusting the two call sites to stay in step.

This is the rate half of the same guard block the sibling modules cover:
``tests/test_dataset_schema_column_names_distinct.py`` (the schema column names)
and ``tests/test_dataset_schema_frame_shape_domain.py`` (the frame shape, whose
premise pin for the behaviour above this module replaces). All three refusals sit
ahead of the same two side effects - the lazy lerobot import, so one caller
mistake reports identically on a minimal install (which is why every refusal test
here runs without the extra), and the on-disk target that ``overwrite=True``
removes.
"""

import ast
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots import dataset_recorder as recorder_mod
from strands_robots.simulation.recording import dataset_recording_option_error
from strands_robots.utils import positive_whole_number_error

# Reached through the module alias rather than a second from-import: the
# guard-placement tests read the module's own source, and one handle for one
# module keeps that unambiguous.
DatasetRecorder = recorder_mod.DatasetRecorder

# Rates no dataset can be written at. ``positive_whole_number_error`` owns this
# domain; the labels are reused as parametrize ids. ``huge`` is refused with a
# reason of its own (it *is* a positive whole number, just not one a float64
# stands for), so it is listed separately below rather than here.
UNUSABLE_RATES: list[tuple[str, Any]] = [
    ("zero", 0),
    ("negative", -5),
    ("fractional", 2.7),
    ("bool", True),
    ("nan", math.nan),
    ("inf", math.inf),
    ("numeric_string", "30"),
    ("none", None),
    ("list", [30]),
    ("mapping", {"fps": 30}),
]

# Rates the recorder demonstrably honors, which must still reach the dataset.
# ``integral_float`` and ``numpy_int`` are accepted because the shared domain
# accepts them for the facades - see ``TestTheRuleIsTheFacadesRule``.
USABLE_RATES: list[tuple[str, Any]] = [
    ("default", 30),
    ("one", 1),
    ("high", 120),
    ("integral_float", 30.0),
    ("numpy_int", np.int64(30)),
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

    Most tests here pass a value the ``fps: int`` annotation forbids (a bare
    string, a mapping, a ``nan``) because that is precisely the caller mistake
    the runtime guard exists for. Routing them through one ``**kwargs: Any``
    funnel states that intent once instead of scattering per-call type
    suppressions.
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
        raise AssertionError("the lerobot extra was probed before fps was checked")

    monkeypatch.setattr(recorder_mod, "_get_lerobot_dataset_class", _fatal)


class TestUnusableRatesAreRefused:
    """Every rate above is refused, naming the parameter and the method."""

    @pytest.mark.parametrize(("label", "value"), UNUSABLE_RATES, ids=[r[0] for r in UNUSABLE_RATES])
    def test_refused_naming_the_parameter_and_the_method(self, no_lerobot: None, label: str, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", joint_names=["j1"], fps=value)
        text = str(excinfo.value)
        assert "fps" in text, text
        assert "must be a positive whole number" in text, text
        assert "DatasetRecorder.create" in text, text

    def test_a_rate_past_the_float_range_is_refused_for_its_own_reason(self, no_lerobot: None) -> None:
        """``10**400`` *is* a positive whole number, so that text would be false."""
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", joint_names=["j1"], fps=10**400)
        text = str(excinfo.value)
        assert "fps must be within the range of a 64-bit float" in text, text

    def test_a_camera_dataset_is_refused_on_the_same_rate(self, no_lerobot: None) -> None:
        """The rate is not a per-camera property, so a declared camera changes nothing."""
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", camera_keys=["image"], camera_dims={"image": (240, 320)}, fps=math.nan)
        assert "fps must be a positive whole number" in str(excinfo.value)


class TestUsableRatesStillReachTheDataset:
    """The over-reach controls: nothing that could be honored is refused."""

    @pytest.mark.parametrize(("label", "value"), USABLE_RATES, ids=[r[0] for r in USABLE_RATES])
    def test_accepted_and_declared_as_given(
        self, fake_lerobot: type[_FakeLeRobotDataset], label: str, value: Any
    ) -> None:
        _create(repo_id="local/probe", joint_names=["j1"], fps=value)
        assert len(fake_lerobot.calls) == 1
        # Forwarded verbatim: the guard refuses, it does not coerce, so the rate
        # the dataset declares is the one the caller passed.
        assert fake_lerobot.calls[0]["fps"] is value

    def test_the_default_rate_is_inside_the_domain(self, fake_lerobot: type[_FakeLeRobotDataset]) -> None:
        """A default the guard refuses would break every call that omits it."""
        default = inspect.signature(DatasetRecorder.create).parameters["fps"].default
        assert positive_whole_number_error(default, "fps", "DatasetRecorder.create") is None
        _create(repo_id="local/probe", joint_names=["j1"])
        assert fake_lerobot.calls[0]["fps"] == default


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
            _create(repo_id="local/probe", root=str(root), joint_names=["j1"], fps=2.7, overwrite=True)
        except ValueError as exc:
            raised = exc

        # Assert the surviving target first: that is the consequence that matters,
        # and it is what a refusal arriving after ``_prepare_create_target`` would
        # already have destroyed.
        assert (root / "meta" / "info.json").is_file(), "the refused call deleted the dataset"
        assert fake_lerobot.calls == [], "the refused call still built a dataset"
        assert raised is not None, "the fractional fps was not refused"
        assert "fps must be a positive whole number" in str(raised), str(raised)

    def test_a_usable_call_does_reach_the_dataset_target(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset]
    ) -> None:
        """The control: the same call at a usable rate is not refused."""
        _create(repo_id="local/probe", root=str(tmp_path / "fresh"), joint_names=["j1"], fps=30)
        assert len(fake_lerobot.calls) == 1


class TestTheRuleIsTheFacadesRule:
    """One rule for the rate, stated once and reached from both surfaces.

    The facades apply it as an error envelope and ``create`` as a ``ValueError``,
    which is a difference in how a refusal is *returned*, not in what is refused.
    A gap either way is a defect: a value the facade accepts and ``create``
    refuses raises out of a method that returns envelopes, and a value the facade
    refuses and ``create`` accepts is a rate one entry point writes and the other
    will not.
    """

    @pytest.mark.parametrize(
        ("label", "value"),
        UNUSABLE_RATES + USABLE_RATES,
        ids=[r[0] for r in UNUSABLE_RATES] + [f"usable_{r[0]}" for r in USABLE_RATES],
    )
    def test_both_surfaces_reach_the_same_verdict(
        self, fake_lerobot: type[_FakeLeRobotDataset], label: str, value: Any
    ) -> None:
        facade_refuses = dataset_recording_option_error("start_recording", value) is not None
        try:
            _create(repo_id="local/probe", joint_names=["j1"], fps=value)
            create_refuses = False
        except ValueError:
            create_refuses = True
        assert create_refuses is facade_refuses, f"verdict differs for {value!r}"

    def test_the_message_is_the_shared_one_verbatim(self, no_lerobot: None) -> None:
        """No second wording for the same rule - only the surface name differs."""
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", joint_names=["j1"], fps=2.7)
        assert str(excinfo.value) == positive_whole_number_error(2.7, "fps", "DatasetRecorder.create")

    def test_the_facade_states_the_same_rule_against_its_own_name(self) -> None:
        """The envelope is the same text under the facade's method name."""
        error = dataset_recording_option_error("start_recording", 2.7)
        assert error is not None
        assert error["content"][0]["text"] == positive_whole_number_error(2.7, "fps", "start_recording")


class TestGuardPlacement:
    """``create`` checks the rate before either of its two side effects."""

    def test_the_guard_delegates_to_the_shared_domain_with_the_method_name(self) -> None:
        """A future edit cannot restate the domain or mislabel the surface."""
        source = inspect.getsource(DatasetRecorder.create)
        tree = ast.parse(inspect.cleandoc(source).replace("@classmethod\n", "", 1))
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)

        call = next(
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "positive_whole_number_error"
        )
        assert [ast.unparse(arg) for arg in call.args] == ["fps", "'fps'", "'DatasetRecorder.create'"]

    def test_the_rate_is_checked_before_the_lerobot_probe_and_the_target(self) -> None:
        """Source order: names, then the shape, then the rate, then any work."""
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

        watched = {
            "name_list_error",
            "_frame_shape_error",
            "positive_whole_number_error",
            "_get_lerobot_dataset_class",
        }
        index_of = {name: i for i, stmt in enumerate(func.body) for name in called_names(stmt) if name in watched}
        assert index_of["_frame_shape_error"] < index_of["positive_whole_number_error"], index_of
        assert index_of["positive_whole_number_error"] < index_of["_get_lerobot_dataset_class"], index_of


class TestNeighbouringSurfacesStayOutOfScope:
    """Premise pins, so a follow-up is not closed by accident."""

    def test_resume_declares_no_rate_of_its_own(self) -> None:
        """``resume`` inherits the schema from disk, so there is no rate to guard."""
        assert "fps" not in inspect.signature(DatasetRecorder.resume).parameters

    def test_the_rate_is_not_compared_against_a_capture_frequency_here(
        self, fake_lerobot: type[_FakeLeRobotDataset]
    ) -> None:
        """A rate a rollout is not capturing at is the facades' check, not this one.

        ``_validate_recording_start_rate`` and ``dataset_rate_mismatch_error``
        compare a *usable* rate against the frequency a rollout runs at, which
        the recorder cannot see. So ``create`` accepts any usable rate on its own
        terms; only the domain moved here.
        """
        _create(repo_id="local/probe", joint_names=["j1"], fps=1)
        assert len(fake_lerobot.calls) == 1
