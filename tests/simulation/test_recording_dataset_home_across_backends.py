"""Every backend's ``start_recording`` resolves the dataset dir the same way.

``start_recording`` is reimplemented per backend, and each copy has to answer the
same question before it touches the disk: which directory does ``repo_id`` /
``root`` name? :func:`~strands_robots.dataset_recorder.resolve_dataset_dir` is
the one answer -- it is what ``DatasetRecorder.create`` itself resolves with, so
a backend that computes its own is deciding where a dataset lives while the
recorder writes somewhere else.

The Newton backend used to hand-roll the resolution. Its first two branches
matched the resolver and the third hard-coded ``~/.cache/huggingface/lerobot``,
so it ignored ``$HF_LEROBOT_HOME`` -- the override LeRobot honours and the only
way to relocate the dataset home. Measured on a scene whose home is relocated,
with ``repo_id="user/ds"`` and ``root=None``:

* ``last_dataset_root`` named the stale ``~/.cache`` path while the MuJoCo
  backend named the configured one, for byte-identical arguments;
* ``overwrite=True`` removed the dataset at the stale path -- one the call never
  addressed and which lives outside the configured home -- and left the
  addressed one for ``create()`` to remove;
* the resume probe missed an existing dataset in the configured home, so
  appending an episode dead-ended in ``FileExistsError`` telling the caller to
  use ``DatasetRecorder.resume()`` instead -- i.e. to bypass the method they
  called;
* ``last_dataset_root``, which ``stop_recording(bucket=...)`` syncs and
  ``verify_dataset_episodes`` reads once the recorder has been dropped, named a
  directory the session never wrote to.

The behavioural half drives the Newton engine through a hand-built ``SimWorld``
(the resolution runs before any solver call) with a stub recorder, so it needs
neither Newton nor lerobot installed. The structural half pins the property for
the two backends whose simulators cannot be driven here, and for any backend
added later.

A third half covers the same contract's *documentation*. Making the override
load-bearing on all three backends left every ``start_recording`` docstring
describing ``root`` as overriding "the repo_id cache-path resolution" without
naming the cache or that it is configurable -- so the resolution had one owner in
code and three partial restatements in prose, which is the arrangement that
drifted in the first place.

That documentation half then generalized past the dataset directory, because
``start_recording`` applies a second multi-level contract it does not own:
``add_frame`` decides a frame's ``task``, and the session's ``task=`` is only the
middle level of it. Two backends described that parameter as the value "recorded
with every frame" -- which is what it is *not*, since every rollout hook passes
``run_policy(instruction=...)`` as the frame task and a non-empty instruction
wins -- and the third documented neither ``task`` nor ``push_to_hub`` at all. The
terminal level, the literal ``"untitled"``, was named nowhere: record with neither
set and every frame carries it, which is a constant instruction for the
language-conditioned policies this repo targets. So the assertions below cover
both contracts: what a caller is told about *where* a recording lands, and about
*what* each of its frames is annotated with.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import strands_robots.dataset_recorder as dr
from strands_robots.dataset_recorder import resolve_dataset_dir
from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_JOINTS = ["Rotation", "Pitch", "Elbow"]


def _engine() -> NewtonSimEngine:
    """A Newton engine bound to a hand-built world, without the Warp stack.

    Mirrors ``tests/simulation/newton/test_dataset_recording.py``: the recording
    lifecycle touches no physics, so the engine is built via ``__new__`` and
    given only the attributes ``start_recording`` reads.
    """
    world = SimWorld()
    world.robots["so100"] = SimRobot(
        name="so100", urdf_path="so100.xml", data_config="so100", joint_names=list(_JOINTS)
    )
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = object()  # non-None sentinel: "world created"
    engine.default_width = 64
    engine.default_height = 48
    return engine


class _StubRecorder:
    """Stands in for ``DatasetRecorder`` and records which path was taken.

    ``create`` and ``resume`` are the two outcomes the dataset-dir resolution
    selects between, so the calls are recorded rather than the objects.
    """

    calls: list[str] = []

    @classmethod
    def create(cls, **kwargs: object) -> object:
        cls.calls.append("create")
        return object()

    @classmethod
    def resume(cls, **kwargs: object) -> object:
        cls.calls.append("resume")
        return object()


@pytest.fixture
def relocated_home(monkeypatch, tmp_path):
    """Point the shared resolver's dataset home at ``tmp_path``.

    Patches ``_lerobot_home`` rather than the environment because lerobot reads
    ``HF_LEROBOT_HOME`` into a module constant at import time, so setting the
    variable here would not move an already-imported home -- and lerobot need not
    be installed at all. This is the same seam
    ``tests/simulation/mujoco/test_recording_paths.py`` pins the MuJoCo backend
    through.
    """
    home = tmp_path / "relocated" / "lerobot"
    monkeypatch.setattr(dr, "_lerobot_home", lambda: home)
    monkeypatch.setattr(dr, "lerobot_dataset_import_error", lambda: None)
    monkeypatch.setattr(dr, "has_lerobot_dataset", lambda: True)
    _StubRecorder.calls = []
    monkeypatch.setattr(dr, "DatasetRecorder", _StubRecorder)
    return home


@pytest.fixture
def contained_user_home(monkeypatch, tmp_path):
    """Redirect ``Path.home()`` into ``tmp_path`` for the duration of a test.

    The hard-coded default the fix removes is spelled relative to the user's home
    directory, so a test that wants to observe what it touched has to move that
    home -- reading or writing the developer's real
    ``~/.cache/huggingface/lerobot`` is not an option. Patched on ``pathlib.Path``
    itself because the resolution under test calls ``Path.home()`` directly.
    """
    home = tmp_path / "user_home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _seed_dataset(directory: Path) -> None:
    """Make ``directory`` look like a real LeRobotDataset (it has ``meta/``)."""
    (directory / "meta").mkdir(parents=True)
    (directory / "meta" / "info.json").write_text("{}")


class TestNewtonResolvesTheConfiguredDatasetHome:
    """The resolved directory is the one the recorder will actually write to."""

    def test_a_namespaced_repo_id_resolves_under_the_configured_home(self, relocated_home):
        """The stashed root is the resolver's answer, not the ``~/.cache`` default.

        ``last_dataset_root`` is the only record of where a finished dataset
        lives once ``stop_recording`` drops the recorder, so a stale value sends
        ``stop_recording(bucket=...)`` and ``verify_dataset_episodes`` at a
        directory this session never wrote to.
        """
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=None, fps=30)

        assert result["status"] == "success", result
        assert engine._world._backend_state["last_dataset_root"] == str(relocated_home / "user" / "ds")

    @pytest.mark.parametrize(
        "repo_id",
        ["user/ds", "bare_local", "./relative_ds", "/tmp/absolute_ds"],
    )
    def test_every_repo_id_shape_agrees_with_the_shared_resolver(self, relocated_home, repo_id):
        """Not just the relocated branch: no branch may disagree with the resolver.

        The hand-rolled copy matched on the path-shaped ``repo_id`` branches and
        diverged only on the home-relative one, which is exactly why reading the
        two side by side did not surface it.
        """
        engine = _engine()

        result = engine.start_recording(repo_id=repo_id, root=None, fps=30)

        assert result["status"] == "success", result
        assert engine._world._backend_state["last_dataset_root"] == str(resolve_dataset_dir(repo_id, None))

    def test_an_explicit_root_still_wins(self, relocated_home, tmp_path):
        """``root=`` is used verbatim, so relocating the home cannot override it."""
        explicit = tmp_path / "explicit"
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=str(explicit), fps=30)

        assert result["status"] == "success", result
        assert engine._world._backend_state["last_dataset_root"] == str(explicit)


class TestOverwriteRemovesOnlyTheAddressedDataset:
    """``overwrite=True`` deletes a dataset, so it must delete the right one."""

    def test_overwrite_clears_the_dataset_inside_the_configured_home(self, relocated_home):
        addressed = relocated_home / "user" / "ds"
        _seed_dataset(addressed)
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=None, fps=30, overwrite=True)

        assert result["status"] == "success", result
        assert not addressed.exists(), "the dataset the call addressed survived overwrite=True"

    def test_overwrite_leaves_an_identically_named_dataset_under_the_default_home(
        self, relocated_home, contained_user_home
    ):
        """The destructive half, and the reason this is a bug rather than a mismatch.

        A dataset at the same ``repo_id`` under the ``~/.cache`` default is not
        the one ``overwrite=True`` was asked to replace once the home has been
        moved. Removing it destroys recorded episodes at a path the call never
        named, under ``status="success"``.

        ``Path.home`` is contained for this one assertion because the pre-fix
        path is only reachable through it; asserting against the real
        ``~/.cache/huggingface/lerobot`` is the one thing this test must not do.
        """
        bystander = contained_user_home / ".cache" / "huggingface" / "lerobot" / "user" / "ds"
        _seed_dataset(bystander)
        _seed_dataset(relocated_home / "user" / "ds")
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=None, fps=30, overwrite=True)

        assert result["status"] == "success", result
        assert (bystander / "meta" / "info.json").exists(), (
            "overwrite=True removed a dataset under the default home, which this call did not address"
        )


class TestAnExistingDatasetIsResumedNotRecreated:
    """The resume probe reads the resolved dir, so it must read the right one."""

    def test_an_existing_dataset_in_the_configured_home_is_resumed(self, relocated_home):
        """Missing it is not a slower path but a dead end.

        ``create()`` refuses a directory holding a ``meta/`` with a
        ``FileExistsError`` naming ``overwrite=True`` (which discards the
        recorded episodes) and ``DatasetRecorder.resume()`` (which bypasses this
        method), so appending an episode had no route through the public API.
        """
        _seed_dataset(relocated_home / "user" / "ds")
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=None, fps=30)

        assert result["status"] == "success", result
        assert _StubRecorder.calls == ["resume"], f"expected an append, got {_StubRecorder.calls}"

    def test_a_fresh_repo_id_is_still_created(self, relocated_home):
        """The mirror: nothing on disk stays a ``create``, not a resume."""
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=None, fps=30)

        assert result["status"] == "success", result
        assert _StubRecorder.calls == ["create"], f"expected a fresh dataset, got {_StubRecorder.calls}"

    def test_create_refuses_the_dataset_the_stale_probe_used_to_miss(self, relocated_home):
        """Pins the dead end itself, so the resume assertion above has a stated cost.

        This is the error a caller reached before the fix: the probe reported
        "nothing on disk", ``create()`` disagreed, and the message pointed away
        from ``start_recording``.
        """
        addressed = resolve_dataset_dir("user/ds", None)
        _seed_dataset(addressed)

        with pytest.raises(FileExistsError, match="already exists"):
            dr._prepare_create_target(addressed, overwrite=False)


_START_RECORDING_BACKENDS = (
    "strands_robots/simulation/mujoco/recording.py",
    "strands_robots/simulation/isaac/recording.py",
    "strands_robots/simulation/newton/recording.py",
)

#: This repository, located from this file. Every path below is joined onto it:
#: a relative literal resolves against the working directory, so the reads here
#: raised ``FileNotFoundError`` and the sweep above graded nothing at all when
#: the suite ran from anywhere but the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_TREE = _REPO_ROOT / "docs"


def _method_node(module: str, method: str) -> ast.FunctionDef:
    """The ``def method`` node in ``module``, or fail naming what was not found.

    Both readers below need the same lookup, and a scanner that silently found
    no method would assert nothing at all - so a missing name fails the test
    rather than yielding an empty set.
    """
    tree = ast.parse((_REPO_ROOT / module).read_text())
    found = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == method),
        None,
    )
    if found is None:
        pytest.fail(f"{method} not found in {module}")
    return found


def _called_names(module: str, method: str) -> set[str]:
    """Names of every plain ``f(...)`` call made inside ``module::method``."""
    node = _method_node(module, method)
    return {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _string_constants(module: str, method: str) -> set[str]:
    """Every string literal appearing inside ``module::method``."""
    node = _method_node(module, method)
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


class TestNoDatasetRootResolutionDrifts:
    """Pinned structurally: the Isaac and Newton simulators cannot be driven here.

    Behavioural coverage above can only reach the backend whose recording
    lifecycle runs solver-free. These two assertions hold the property for the
    others, and for a backend added later -- which is how the hand-rolled copy
    survived: nothing said the resolution had one owner.
    """

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    def test_every_backend_start_recording_uses_the_shared_resolver(self, module):
        assert "resolve_dataset_dir" in _called_names(module, "start_recording"), (
            f"{module}::start_recording resolves the dataset dir without resolve_dataset_dir"
        )

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    def test_no_backend_spells_the_dataset_home_itself(self, module):
        """A literal home component is the shape the drift took.

        ``resolve_dataset_dir`` reads the home from lerobot's own constant so the
        ``HF_LEROBOT_HOME`` override is honoured; a backend that spells any part
        of the default path has pinned it instead, which is what silently
        ignored the override.
        """
        literals = _string_constants(module, "start_recording")
        assert not literals & {".cache", "huggingface", "lerobot"}, (
            f"{module}::start_recording hard-codes the dataset home instead of resolving it"
        )


def _docstring(module: str, method: str) -> str:
    """The docstring of ``module::method``, or fail if it has none.

    Read separately from :func:`_string_constants` because that reader answers
    "what does the code spell" while these assertions answer "what does the
    caller get told" - a method whose docstring were removed entirely would
    satisfy an emptied-set check silently.
    """
    doc = ast.get_docstring(_method_node(module, method))
    if not doc:
        pytest.fail(f"{method} in {module} has no docstring")
    return doc


def _arg_entry(module: str, method: str, param: str) -> str:
    """The ``param:`` entry of ``module::method``'s Args block, or ``""`` if absent.

    Read per parameter rather than over the whole docstring, because the claims
    below are about what *this* entry says: a docstring-wide search would let a
    correct sentence about some other parameter satisfy an assertion about
    ``task``, and would let a wrong claim in the ``task`` entry be excused by a
    right one elsewhere. An entry runs from its ``param:`` line to the next line
    indented no deeper, which is how these Args blocks are already written.
    """
    lines = _docstring(module, method).splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith(f"{param}:"):
            indent = len(line) - len(line.lstrip())
            entry = [line.strip()]
            for follow in lines[index + 1 :]:
                if not follow.strip():
                    continue
                if len(follow) - len(follow.lstrip()) <= indent:
                    break
                entry.append(follow.strip())
            return " ".join(entry)
    return ""


class TestStartRecordingDocumentsWhereTheDatasetLands:
    """The resolution has one owner in code; its documentation needs one too.

    The scanners above hold every backend to the shared resolver, which settles
    *behaviour*. They say nothing about what a caller reading ``start_recording``
    is told, and the answer was nothing: all three docstrings described ``root``
    as overriding "the repo_id cache-path resolution" without naming which cache
    or that it is configurable, the MuJoCo one documented no ``repo_id`` at all,
    and ``$HF_LEROBOT_HOME`` appeared only in inline code comments a caller never
    reads. So the override that #1863 made load-bearing on all three backends was
    undiscoverable from the API surface, and three partial restatements of one
    contract were free to drift apart again.

    Each assertion pins the cross-reference rather than the prose, so the rules
    keep a single owner: restating them per backend is what drifted.
    """

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    def test_every_backend_points_at_the_shared_resolver(self, module):
        assert "resolve_dataset_dir" in _docstring(module, "start_recording"), (
            f"{module}::start_recording does not tell the caller which resolver decides the dataset directory"
        )

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    def test_every_backend_names_the_home_override(self, module):
        """The one thing a caller cannot find out any other way.

        ``resolve_dataset_dir`` reads the home from lerobot's own constant, so
        the environment variable is the only way to move a recording - and a
        cross-reference alone leaves the reader to go find that out.
        """
        assert "HF_LEROBOT_HOME" in _docstring(module, "start_recording"), (
            f"{module}::start_recording never names HF_LEROBOT_HOME, so the dataset home reads as fixed"
        )

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    def test_every_backend_documents_both_parameters_that_select_the_directory(self, module):
        """``repo_id`` and ``root`` are the two inputs to the resolution.

        MuJoCo's Args block documented ``fps`` / ``root`` / ``overwrite`` /
        ``vcodec`` / ``cameras`` and omitted the parameter that names the
        dataset, so a cross-reference on ``root`` alone would still leave the
        default path unexplained on that backend.
        """
        doc = _docstring(module, "start_recording")
        missing = [param for param in ("repo_id:", "root:") if param not in doc]
        assert not missing, f"{module}::start_recording documents no {', '.join(missing)} entry"

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    def test_every_backend_points_at_the_shared_create_vs_resume_owner(self, module):
        """The same one-owner rule for the other contract documented only on a helper.

        ``_prepare_dataset_target`` decides four outcomes for an existing target
        (resume, clear-empty, wipe-on-overwrite, refuse-non-dataset). Two
        backends described it as "wipe and recreate ... instead of appending",
        which names two of the four and reads as though an empty ``root`` were
        an error.
        """
        assert "_prepare_dataset_target" in _docstring(module, "start_recording"), (
            f"{module}::start_recording restates the create-vs-resume outcomes instead of citing their owner"
        )

    def test_no_doc_page_sends_a_reader_to_a_dataset_home_nothing_writes(self):
        """``~/.strands_robots/datasets/`` is not where a recording lands.

        ``~/.strands_robots`` is the renders / mesh-audit / scene-cache home; no
        code writes a dataset under it. Three doc passages named it as the
        default recording location anyway - the quick-start comment, the
        ``DatasetRecorder`` snippet, and an annotation workflow that handed
        ``lerobot-annotate --root`` a path the preceding ``start_recording``
        never wrote to, so the documented sequence could not work as written.
        The three were independent copies, which is why fixing one left two, so
        the property is asserted over the whole doc tree rather than per page.
        """
        pages = sorted(_DOCS_TREE.rglob("*.md"))
        assert len(pages) > 20, (
            f"the sweep read {len(pages)} pages, so it graded almost nothing - it is rooted at "
            f"{_DOCS_TREE}, which is not this repository's doc tree"
        )
        offenders = [
            path.relative_to(_REPO_ROOT).as_posix() for path in pages if "strands_robots/datasets" in path.read_text()
        ]
        assert not offenders, (
            f"{', '.join(offenders)} names ~/.strands_robots/datasets as a dataset location; "
            "recordings land under $HF_LEROBOT_HOME (see resolve_dataset_dir)"
        )


_PER_FRAME_CLAIMS = (
    "recorded with every frame",
    "recorded with each frame",
    "recorded on every frame",
    "written to every frame",
    "written with every frame",
    "applied to every frame",
    "used for every frame",
)


class TestStartRecordingDocumentsTheTaskThatLandsOnAFrame:
    """The other multi-level contract ``start_recording`` applies and does not own.

    ``add_frame`` is the whole rule -- ``task or self.default_task or "untitled"``
    -- and ``start_recording(task=...)`` feeds only the middle term, via
    ``create(task=...)`` / ``resume(task=...)``. Which term supplies the value a
    dataset is annotated with therefore depends on an argument to a *different*
    method: every rollout hook passes ``run_policy(instruction=...)`` as the frame
    task, and that argument defaults to empty.

    Two of the three backends described the parameter as the value "recorded with
    every frame", which asserts it is the per-frame term rather than the middle
    one. That failure mode is quiet in a way an omission is not: a caller who sets
    ``task="pick up the red cube"`` and then runs
    ``run_policy(..., instruction="place the cube in the bin")`` gets every frame
    annotated with the second string, and both strings are plausible, so the
    dataset reads as correctly annotated. The third backend documented neither
    ``task`` nor ``push_to_hub``.

    The assertions pin the two facts a caller cannot infer plus the claim that is
    wrong, rather than the prose, so the rule keeps the single owner #1865
    established for the directory resolution.
    """

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    @pytest.mark.parametrize("param", ["task", "push_to_hub"])
    def test_every_backend_documents_the_two_remaining_parameters(self, module, param):
        """``start_recording`` takes eight parameters on all three backends.

        Six were documented everywhere after #1865 completed the MuJoCo block
        with ``repo_id``; these are the two it deferred, and MuJoCo's Args block
        is where they were missing. Asserted across all three so the next backend
        cannot ship a seven-of-eight block either.
        """
        assert _arg_entry(module, "start_recording", param), f"{module}::start_recording documents no {param} entry"

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    def test_every_task_entry_names_the_terminal_fallback(self, module):
        """Set neither this nor an instruction and every frame reads ``"untitled"``.

        That is the silent case, and it was documented nowhere: the task string is
        the conditioning signal for every language-conditioned policy this repo
        targets, so a dataset recorded with neither set trains against a constant
        instruction. ``start_recording``'s own success text hints at the coupling
        with "(set per policy)", which names the override but not where an unset
        instruction actually lands.
        """
        entry = _arg_entry(module, "start_recording", "task")
        assert "untitled" in entry, (
            f"{module}::start_recording's task entry never names the untitled fallback, "
            "so a dataset annotated entirely with it reads as unexplained"
        )

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    def test_every_task_entry_names_what_overrides_it(self, module):
        """The override is an argument to another method, so it cannot be inferred.

        ``run_policy(instruction=...)`` is what every rollout hook passes as the
        frame task; nothing about ``start_recording``'s own signature says that a
        rollout can displace the value it was given.
        """
        entry = _arg_entry(module, "start_recording", "task")
        assert "instruction" in entry, (
            f"{module}::start_recording's task entry never names run_policy(instruction=...), "
            "so the value it documents reads as the one that wins"
        )

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    def test_no_task_entry_claims_it_is_the_per_frame_value(self, module):
        """The non-vacuous one: this fails on two backends as they stood.

        #1865's ``documents_both_parameters`` check passed on Isaac and Newton
        precisely because their blocks were already complete -- completeness is
        not accuracy, and a wording that is present but wrong needs its own
        assertion. Phrasings rather than one literal string, so the claim cannot
        come back reworded.
        """
        entry = _arg_entry(module, "start_recording", "task").lower()
        claimed = [claim for claim in _PER_FRAME_CLAIMS if claim in entry]
        assert not claimed, (
            f"{module}::start_recording's task entry claims to be {claimed[0]!r}; "
            "it is the middle of add_frame's chain, overridden by run_policy(instruction=...)"
        )

    def test_the_owner_states_the_whole_chain(self):
        """A cross-reference is only worth following if the target answers it.

        The three entries above cite
        ``DatasetRecorder.add_frame`` for the precedence, which is the same
        arrangement #1865 used for ``resolve_dataset_dir`` -- with one difference:
        that resolver already documented its own rules, while ``add_frame``'s task
        entry read "uses default if None" and named neither the session default it
        means nor the terminal fallback. So the owner is pinned too, or the
        cross-references point at nothing.
        """
        entry = _arg_entry("strands_robots/dataset_recorder.py", "add_frame", "task")
        missing = [term for term in ("default_task", "untitled", "instruction") if term not in entry]
        assert not missing, (
            f"DatasetRecorder.add_frame's task entry names no {', '.join(missing)} - "
            "the backends cite it for the precedence it is supposed to own"
        )
