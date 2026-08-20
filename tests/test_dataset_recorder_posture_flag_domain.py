"""The recorder's creation API checks its posture flags, as its siblings do.

``DatasetRecorder.create`` takes three flags that select a *posture* rather than
scaling a quantity - ``use_videos``, ``streaming_encoding`` and ``overwrite`` -
and ``resume`` takes the second of them. All four were read by truthiness. Every
non-empty string is truthy, so ``"false"`` / ``"no"`` / ``"off"`` / ``"0"`` - the
spellings a caller reaches for when opting out - selected the branch being opted
*out* of, and ``None`` / ``0`` / ``""`` / ``[]`` took the other branch without
ever being a declared spelling of it.

Measured on ``ba04dd21`` against a dataset holding one recorded episode:

=============================  ==============  ==========================
call                           ``create()``    consequence
=============================  ==============  ==========================
``overwrite=True``             success         1 episode -> 0 (as asked)
``overwrite=False``            FileExistsError 1 episode -> 1 (as asked)
``overwrite="false"``          success         1 episode -> **0**
``overwrite="no"``/``"off"``   success         1 episode -> **0**
``overwrite="0"``              success         1 episode -> **0**
``use_videos=False``           success         schema ``dtype="image"``
``use_videos="false"``         success         schema ``dtype="video"``
=============================  ==============  ==========================

``overwrite`` is a confirmation gate in front of a delete: read as True it
reaches :func:`~strands_robots.dataset_recorder._prepare_create_target`, which
``shutil.rmtree``s the target. That function already refuses to clobber a
non-empty *non*-dataset directory, so the one thing it removed without asking was
a real LeRobotDataset - the recorded episodes the caller meant to keep - and
``create`` returned a working recorder throughout. ``use_videos`` writes its
branch into ``meta/info.json`` as each camera's ``dtype``, fixed for the life of
the dataset, and the same unconverted string then reached LeRobot's own boolean
parameter, as ``streaming_encoding="false"`` did on both entry points.

**Three of the four posture-flag surfaces in this module already check.**
:meth:`~strands_robots.dataset_recorder.DatasetRecorder.push_to_hub` checks
``private``, :func:`~strands_robots.dataset_recorder.sync_dataset_to_bucket`
checks its three, and every backend's ``start_recording`` facade checks the
``overwrite`` it forwards *here*
(:func:`~strands_robots.simulation.recording.dataset_recording_posture_error`).
The documented direct creation API was the one left reading them by truthiness,
so the facade and the method it forwards to disagreed about which values are
usable - the disagreement the ``fps`` guard in the same block exists to prevent.

This is the posture half of that guard block; the numeric half is covered by
``tests/test_dataset_recorder_fps_domain.py`` (the rate) and
``tests/test_dataset_schema_column_names_distinct.py`` (the column names). The
domains are inverses: the numeric ones reject ``bool`` because an ``int``
subclass would pass as a silent ``1``, and this one requires the boolean they
turn away, so ``TestTheNumericSiblingsKeepTheInverseDomain`` pins that both hold
in the same signature.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path
from typing import Any

import pytest

from strands_robots import dataset_recorder as recorder_mod
from strands_robots.simulation.recording import dataset_recording_posture_error
from strands_robots.utils import boolean_flag_error

# Reached through the module alias rather than a second from-import: the
# guard-placement tests read the module's own source, and one handle for one
# module keeps that unambiguous.
DatasetRecorder = recorder_mod.DatasetRecorder

# The posture flags each creation entry point declares. Derived from the live
# signatures in ``TestEveryDeclaredPostureFlagIsChecked`` rather than trusted
# here, so a fourth flag added later is covered without an edit to this list.
CREATION_FLAGS: dict[str, tuple[str, ...]] = {
    "create": ("use_videos", "streaming_encoding", "overwrite"),
    "resume": ("streaming_encoding",),
}

# Values that are not booleans. ``boolean_flag_error`` owns this domain - the
# labels are only parametrize ids. The four string spellings are the ones an
# opt-out reaches for; the rest are the falsy values that took a branch they
# were never a declared spelling of.
UNHONORABLE: list[tuple[str, Any]] = [
    ("string_false", "false"),
    ("string_no", "no"),
    ("string_off", "off"),
    ("string_zero", "0"),
    ("string_true", "true"),
    ("int_zero", 0),
    ("int_one", 1),
    ("none", None),
    ("empty_string", ""),
    ("empty_list", []),
    ("float_one", 1.0),
    ("nan", math.nan),
]

HONORABLE: list[tuple[str, Any]] = [("true", True), ("false", False)]


class _FakeLeRobotDataset:
    """Stand-in for ``LeRobotDataset``, recording what reached it.

    ``streaming_encoding`` is declared explicitly rather than left to
    ``**kwargs``: both entry points forward it only ``if "streaming_encoding" in
    inspect.signature(LeRobotDatasetCls.create).parameters``, because the
    parameter is newer than the oldest supported LeRobot. A stand-in that
    accepted it only through ``**kwargs`` would fail that check and record the
    flag as never forwarded, which reads as the guard dropping it.
    """

    create_calls: list[dict[str, Any]] = []
    resume_calls: list[dict[str, Any]] = []

    def __init__(self, features: dict[str, Any]) -> None:
        self.repo_id = "local/fake"
        self.features = features
        self.meta = None

    @classmethod
    def create(cls, *, streaming_encoding: bool = True, **kwargs: Any) -> _FakeLeRobotDataset:
        cls.create_calls.append({**kwargs, "streaming_encoding": streaming_encoding})
        return cls(kwargs.get("features", {}))

    @classmethod
    def resume(cls, *, streaming_encoding: bool = True, **kwargs: Any) -> _FakeLeRobotDataset:
        cls.resume_calls.append({**kwargs, "streaming_encoding": streaming_encoding})
        return cls({})


def _create(**kwargs: Any) -> DatasetRecorder:
    """Call ``create`` with keywords the ``bool`` annotations disallow.

    Every unhonorable value here is one the annotation forbids, which is
    precisely the caller mistake the runtime guard exists for. One
    ``**kwargs: Any`` funnel states that intent once instead of scattering
    per-call type suppressions.
    """
    return DatasetRecorder.create(**kwargs)


def _resume(**kwargs: Any) -> DatasetRecorder:
    """``resume`` counterpart of :func:`_create`."""
    return DatasetRecorder.resume(**kwargs)


def _existing_dataset(root: Path, *, episodes: int = 1, frames: int = 4) -> Path:
    """Write the on-disk marks that make ``root`` an existing LeRobotDataset.

    ``_prepare_create_target`` decides what to do from the presence of ``meta/``,
    so the recorded contents only have to be readable back: what matters is
    whether they still exist after a refused call.
    """
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"fps": 30, "total_episodes": episodes, "total_frames": frames}),
        encoding="utf-8",
    )
    (root / "data").mkdir()
    (root / "data" / "episode_000000.parquet").write_bytes(b"recorded")
    return root


def _survey(root: Path) -> dict[str, Any]:
    """What a caller would still find at *root*."""
    info = root / "meta" / "info.json"
    if not info.is_file():
        return {"dataset": False, "episodes": None, "parquet": 0}
    meta = json.loads(info.read_text(encoding="utf-8"))
    return {
        "dataset": True,
        "episodes": meta.get("total_episodes"),
        "parquet": len(list(root.rglob("*.parquet"))),
    }


@pytest.fixture
def fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLeRobotDataset]:
    """Route both entry points onto a fake dataset class - no lerobot extra."""
    _FakeLeRobotDataset.create_calls = []
    _FakeLeRobotDataset.resume_calls = []
    monkeypatch.setattr(recorder_mod, "_get_lerobot_dataset_class", lambda: _FakeLeRobotDataset)
    return _FakeLeRobotDataset


@pytest.fixture
def no_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make reaching the lazy lerobot import fatal.

    A refusal must be decided before the dataset extra is probed, so the same
    caller mistake is reported the same way on a minimal install.
    """

    def _fatal() -> Any:
        raise AssertionError("the lerobot extra was probed before the posture flags were checked")

    monkeypatch.setattr(recorder_mod, "_get_lerobot_dataset_class", _fatal)


class TestOverwriteNoLongerDeletesTheDatasetItWasOptingOutOf:
    """The destructive flag: a value that is not a boolean must not delete."""

    @pytest.mark.parametrize(
        ("label", "value"),
        [(lbl, val) for lbl, val in UNHONORABLE if lbl.startswith("string_")],
        ids=lambda arg: arg if isinstance(arg, str) else "",
    )
    def test_an_unhonorable_overwrite_leaves_the_episodes_on_disk(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset], label: str, value: Any
    ) -> None:
        root = _existing_dataset(tmp_path / "ds")
        before = _survey(root)
        assert before == {"dataset": True, "episodes": 1, "parquet": 1}, f"premise: {before}"

        raised: Exception | None = None
        try:
            _create(repo_id="local/probe", root=str(root), joint_names=["j1"], fps=30, overwrite=value)
        except ValueError as exc:
            raised = exc

        # The surviving dataset is asserted first: it is the consequence that
        # matters, and it is what a refusal arriving after
        # ``_prepare_create_target`` would already have destroyed.
        assert _survey(root) == before, f"overwrite={value!r} deleted the dataset it was opting out of"
        assert fake_lerobot.create_calls == [], "the refused call still built a dataset"
        assert raised is not None, f"overwrite={value!r} was not refused"
        assert "overwrite must be a boolean" in str(raised), str(raised)

    def test_the_refusal_precedes_the_lerobot_probe(self, no_lerobot: None, tmp_path: Path) -> None:
        """One caller mistake, one report, on every install."""
        with pytest.raises(ValueError, match="overwrite must be a boolean"):
            _create(repo_id="local/probe", root=str(tmp_path / "ds"), joint_names=["j1"], fps=30, overwrite="false")

    def test_overwrite_true_still_replaces_the_target(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset]
    ) -> None:
        """Control: the documented opt-in is unchanged, and it does delete."""
        root = _existing_dataset(tmp_path / "ds")
        _create(repo_id="local/probe", root=str(root), joint_names=["j1"], fps=30, overwrite=True)
        assert _survey(root)["dataset"] is False, "overwrite=True must still replace the target"
        assert len(fake_lerobot.create_calls) == 1

    def test_overwrite_false_still_refuses_without_deleting(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset]
    ) -> None:
        """Control: the documented opt-out keeps its own distinct verdict.

        ``FileExistsError`` names ``overwrite=True`` and ``resume``; the new guard
        must not displace it with a domain complaint about a valid boolean.
        """
        root = _existing_dataset(tmp_path / "ds")
        with pytest.raises(FileExistsError) as excinfo:
            _create(repo_id="local/probe", root=str(root), joint_names=["j1"], fps=30, overwrite=False)
        assert "overwrite=True" in str(excinfo.value), str(excinfo.value)
        assert _survey(root) == {"dataset": True, "episodes": 1, "parquet": 1}


class TestUseVideosNoLongerInvertsTheRecordedSchema:
    """``use_videos`` picks each camera's ``dtype``, which the dataset keeps."""

    def _declared_dtype(self, calls: list[dict[str, Any]]) -> str:
        features = calls[0]["features"]
        return str(features["observation.images.cam"]["dtype"])

    @pytest.mark.parametrize(("expected", "value"), [("image", False), ("video", True)])
    def test_a_boolean_declares_the_schema_it_names(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset], expected: str, value: bool
    ) -> None:
        """Control: both honorable values are unchanged."""
        _create(
            repo_id="local/probe",
            root=str(tmp_path / f"ds_{expected}"),
            joint_names=["j1"],
            camera_keys=["cam"],
            fps=30,
            use_videos=value,
        )
        assert self._declared_dtype(fake_lerobot.create_calls) == expected

    def test_an_unhonorable_use_videos_declares_no_schema_at_all(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset]
    ) -> None:
        """``"false"`` used to declare ``video`` - the branch it asks to skip."""
        with pytest.raises(ValueError, match="use_videos must be a boolean"):
            _create(
                repo_id="local/probe",
                root=str(tmp_path / "ds"),
                joint_names=["j1"],
                camera_keys=["cam"],
                fps=30,
                use_videos="false",
            )
        assert fake_lerobot.create_calls == [], "a schema was declared from a value that is not a boolean"


class TestBothCreationEntryPointsCheckTheSameFlag:
    """``streaming_encoding`` is forwarded by both, so both must judge it alike."""

    @pytest.mark.parametrize(("label", "value"), UNHONORABLE, ids=[lbl for lbl, _ in UNHONORABLE])
    def test_neither_entry_point_forwards_an_unhonorable_value(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset], label: str, value: Any
    ) -> None:
        with pytest.raises(ValueError, match="streaming_encoding must be a boolean"):
            _create(
                repo_id="local/probe",
                root=str(tmp_path / f"c_{label}"),
                joint_names=["j1"],
                fps=30,
                streaming_encoding=value,
            )
        with pytest.raises(ValueError, match="streaming_encoding must be a boolean"):
            _resume(
                repo_id="local/probe", root=str(_existing_dataset(tmp_path / f"r_{label}")), streaming_encoding=value
            )
        assert fake_lerobot.create_calls == []
        assert fake_lerobot.resume_calls == []

    @pytest.mark.parametrize(("label", "value"), HONORABLE, ids=[lbl for lbl, _ in HONORABLE])
    def test_a_boolean_still_reaches_both_datasets(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset], label: str, value: bool
    ) -> None:
        """Control: the flag is still forwarded, and as the boolean given."""
        _create(
            repo_id="local/probe",
            root=str(tmp_path / f"c_{label}"),
            joint_names=["j1"],
            fps=30,
            streaming_encoding=value,
        )
        _resume(repo_id="local/probe", root=str(_existing_dataset(tmp_path / f"r_{label}")), streaming_encoding=value)
        assert fake_lerobot.create_calls[0]["streaming_encoding"] is value
        assert fake_lerobot.resume_calls[0]["streaming_encoding"] is value

    def test_the_resume_refusal_precedes_the_lerobot_probe(self, no_lerobot: None, tmp_path: Path) -> None:
        """``resume``'s own ``RuntimeError`` must not displace the domain report."""
        with pytest.raises(ValueError, match="streaming_encoding must be a boolean"):
            _resume(repo_id="local/probe", root=str(tmp_path / "ds"), streaming_encoding="false")


class TestTheDomainIsTheSharedOne:
    """One rule for a posture flag, stated once and reached from every surface."""

    @pytest.mark.parametrize(
        ("label", "value"),
        UNHONORABLE + HONORABLE,
        ids=[lbl for lbl, _ in UNHONORABLE] + [f"honorable_{lbl}" for lbl, _ in HONORABLE],
    )
    def test_create_agrees_with_the_domain_value_by_value(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset], label: str, value: Any
    ) -> None:
        """Parametrized over the domain itself, not a copied spelling list."""
        domain_refuses = boolean_flag_error(value, "overwrite", "DatasetRecorder.create") is not None
        try:
            _create(
                repo_id="local/probe",
                root=str(tmp_path / f"ds_{label}"),
                joint_names=["j1"],
                fps=30,
                overwrite=value,
            )
            create_refuses = False
        except ValueError:
            create_refuses = True
        except FileExistsError:  # pragma: no cover - a fresh target cannot exist
            create_refuses = False
        assert create_refuses is domain_refuses, f"verdict differs from the shared domain for {value!r}"

    def test_the_message_is_the_shared_one_verbatim(self, no_lerobot: None, tmp_path: Path) -> None:
        """No second wording for the same rule - only the surface name differs."""
        with pytest.raises(ValueError) as excinfo:
            _create(repo_id="local/probe", root=str(tmp_path / "ds"), joint_names=["j1"], fps=30, overwrite="false")
        assert str(excinfo.value) == boolean_flag_error("false", "overwrite", "DatasetRecorder.create")

    def test_the_facade_states_the_same_rule_against_its_own_name(self) -> None:
        """``start_recording`` forwards ``overwrite`` here and shares the rule."""
        error = dataset_recording_posture_error("start_recording", "overwrite", "false")
        assert error is not None
        assert error["content"][0]["text"] == boolean_flag_error("false", "overwrite", "start_recording")


class TestTheNumericSiblingsKeepTheInverseDomain:
    """A posture flag needs the boolean the numeric knobs turn away."""

    def test_a_boolean_rate_is_still_refused(self, no_lerobot: None, tmp_path: Path) -> None:
        """``fps=True`` recorded a 1 fps dataset; that guard must still hold."""
        with pytest.raises(ValueError, match="fps must be a positive whole number"):
            _create(repo_id="local/probe", root=str(tmp_path / "ds"), joint_names=["j1"], fps=True)

    def test_a_usable_call_declaring_every_flag_is_not_refused(
        self, tmp_path: Path, fake_lerobot: type[_FakeLeRobotDataset]
    ) -> None:
        """Control: all four knobs stated honorably still build a dataset."""
        _create(
            repo_id="local/probe",
            root=str(tmp_path / "ds"),
            joint_names=["j1"],
            camera_keys=["cam"],
            fps=30,
            use_videos=True,
            streaming_encoding=False,
            overwrite=False,
        )
        assert len(fake_lerobot.create_calls) == 1


def _flags_checked_by(source: str, function: str) -> set[str]:
    """Flag names *function* passes to ``boolean_flag_error`` in *source*.

    Reads the literal name argument, and the iterated tuple of a per-flag loop,
    so the scanner sees either spelling of the same guard.
    """
    checked: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.FunctionDef) and node.name == function):
            continue
        for call in ast.walk(node):
            if not (
                isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "boolean_flag_error"
            ):
                continue
            for arg in call.args[1:]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    checked.add(arg.value)
        for loop in ast.walk(node):
            if not isinstance(loop, ast.For):
                continue
            body = ast.unparse(ast.Module(body=list(loop.body), type_ignores=[]))
            if "boolean_flag_error" not in body:
                continue
            for element in ast.walk(loop.iter):
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    checked.add(element.value)
    return checked


class TestEveryDeclaredPostureFlagIsChecked:
    """The root cause: a ``bool`` parameter here is a flag, so it is checked."""

    @pytest.mark.parametrize("method", sorted(CREATION_FLAGS))
    def test_the_live_signature_declares_the_flags_this_file_covers(self, method: str) -> None:
        """Premise: a fourth flag added later fails this before it is missed."""
        signature = inspect.signature(getattr(DatasetRecorder, method))
        declared = {
            name
            for name, param in signature.parameters.items()
            if param.annotation is bool or param.annotation == "bool"
        }
        assert declared == set(CREATION_FLAGS[method]), (
            f"{method} declares {sorted(declared)}; this file covers {sorted(CREATION_FLAGS[method])}"
        )

    @pytest.mark.parametrize("method", sorted(CREATION_FLAGS))
    def test_each_declared_flag_reaches_the_shared_domain(self, method: str) -> None:
        source = Path(inspect.getfile(DatasetRecorder)).read_text(encoding="utf-8")
        checked = _flags_checked_by(source, method)
        assert set(CREATION_FLAGS[method]) <= checked, (
            f"{method} must check {CREATION_FLAGS[method]} via boolean_flag_error; found {sorted(checked)}"
        )

    def test_the_scanner_detects_an_unchecked_flag(self) -> None:
        """Non-vacuity: a creation method with no guard must be reported."""
        planted = "def create(cls, use_videos=True, overwrite=False):\n    return None\n"
        assert _flags_checked_by(planted, "create") == set()

    def test_the_flags_are_checked_before_the_lerobot_probe(self) -> None:
        """Source order inside ``create``: the guard block, then any work."""
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
            "positive_whole_number_error",
            "boolean_flag_error",
            "_get_lerobot_dataset_class",
            "_prepare_create_target",
        }
        index_of = {name: i for i, stmt in enumerate(func.body) for name in called_names(stmt) if name in watched}
        assert index_of["positive_whole_number_error"] < index_of["boolean_flag_error"], index_of
        assert index_of["boolean_flag_error"] < index_of["_get_lerobot_dataset_class"], index_of
        assert index_of["boolean_flag_error"] < index_of["_prepare_create_target"], index_of


class TestNeighbouringSurfacesStayOutOfScope:
    """Premise pins, so a follow-up is not closed by accident."""

    def test_the_hub_and_bucket_surfaces_already_check_their_own_flags(self) -> None:
        """The in-module precedent this guard follows, not scope it adds."""
        source = Path(inspect.getfile(DatasetRecorder)).read_text(encoding="utf-8")
        assert "private" in _flags_checked_by(source, "push_to_hub")
        bucket = _flags_checked_by(source, "sync_dataset_to_bucket")
        assert {"create", "private", "delete"} <= bucket, sorted(bucket)

    def test_strict_fails_loudly_rather_than_silently(self) -> None:
        """``__init__``'s flag is out of scope, and this records why.

        ``strict`` selects raise-vs-drop on a malformed frame, so a truthy
        non-boolean fails toward *raising* - loud, and recoverable. It is not the
        silent-inversion case the creation flags are, and it is still declared, so
        this pins the boundary rather than the behaviour.
        """
        assert inspect.signature(DatasetRecorder.__init__).parameters["strict"].annotation in (bool, "bool")
        source = Path(inspect.getfile(DatasetRecorder)).read_text(encoding="utf-8")
        assert _flags_checked_by(source, "__init__") == set()

    def test_resume_declares_no_rate_of_its_own(self) -> None:
        """``resume`` inherits the schema from disk, so it has no ``fps`` flag."""
        assert "fps" not in inspect.signature(DatasetRecorder.resume).parameters
