"""Pin the :class:`~strands_robots.training.base.Trainer` base contract.

Two things are pinned here.

The **defaults**. A minimal concrete ``Trainer`` (only the three abstract
members) exercises the inherited behaviour of every non-abstract default, so a
regression in the base contract is caught and not only in a subclass override.
:meth:`Trainer.status` in particular is not a fallback reserved for some future
backend: most shipped providers inherit it, so the default IS their live
behaviour, and every one of them is driven through it below.

The **documented contract**. The ABC's prose is the only place a caller learns
what :meth:`Trainer.train` may hand back, and a caller that reads it as always
terminal renders a completed-run report for a run that has not finished. So the
prose is graded against the providers rather than trusted: the statuses each
registered provider's ``train`` can actually return are read out of the code,
and the ABC must name them. That grading is positive on purpose - it asks the
prose to cover what the code does, so it cannot be satisfied by rewording, and a
provider that starts returning a further status is graded on arrival.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import strands_robots.training as training_pkg
from strands_robots.training import create_trainer, factory, list_trainers
from strands_robots.training.base import Trainer, TrainResult, TrainSpec

_TERMINAL_STATUSES = frozenset({"success", "error"})


def _defining_class(cls: type, member: str) -> type:
    """Return the class in ``cls``'s MRO that defines ``member``."""
    for klass in cls.__mro__:
        if member in klass.__dict__:
            return klass
    raise AssertionError(f"{cls.__name__} resolves no member named {member!r}")


def _provider_classes() -> dict[str, type]:
    """Map every provider this package ships to its concrete ``Trainer`` class.

    Scoped to classes defined under ``strands_robots.training``: the runtime
    registry also accepts a trainer registered by a caller (a sibling test
    registers a throwaway one), and such a class is not part of the ABC's
    provider contract - grading it would make the result depend on which other
    tests ran first.
    """
    classes: dict[str, type] = {}
    for name in sorted(list_trainers()):
        cls = type(create_trainer(name))
        if (cls.__module__ or "").startswith(f"{training_pkg.__name__}."):
            classes[name] = cls
    return classes


def _train_statuses(cls: type) -> set[str]:
    """Statuses a provider's ``train`` can return, read out of its source.

    Walks ``train`` in the module that DEFINES it (an RL trainer inherits its
    ``train`` from a different module than its own), following ``self._helper``
    calls, and collects every ``TrainResult(status="...")`` literal it can
    reach. A submitted-job provider builds its non-terminal result in a polling
    helper rather than in ``train`` itself, so following the calls is what makes
    the measurement see it.
    """
    owner = _defining_class(cls, "train")
    source_file = inspect.getsourcefile(owner)
    assert source_file is not None, f"no source for {owner.__name__}"
    tree = ast.parse(Path(source_file).read_text(encoding="utf-8"), filename=source_file)
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    statuses: set[str] = set()
    seen: set[str] = set()
    todo = ["train"]
    while todo:
        name = todo.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        for node in ast.walk(functions[name]):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "TrainResult":
                for keyword in node.keywords:
                    if keyword.arg == "status" and isinstance(keyword.value, ast.Constant):
                        statuses.add(str(keyword.value.value))
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
                todo.append(func.attr)
    return statuses


def _non_terminal_train_statuses() -> dict[str, set[str]]:
    """Provider -> the non-terminal statuses its ``train`` can hand back."""
    found: dict[str, set[str]] = {}
    for name, cls in _provider_classes().items():
        extra = _train_statuses(cls) - _TERMINAL_STATUSES
        if extra:
            found[name] = extra
    return found


def _names_the_status_value(doc: str, status: str) -> bool:
    """Is ``status`` presented as a status VALUE, not just used as an English word?

    ``TrainResult`` documents its own statuses as ``"success"`` | ``"running"`` |
    ``"error"``, and the ABC writes every other status it names the same way, so
    the double-backtick form is what distinguishes naming the value from a
    sentence that happens to contain the word (prose about "a still-running job"
    names no return value).
    """
    return f"``{status}``" in doc


def _sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping common abbreviations intact."""
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    merged: list[str] = []
    for part in parts:
        if merged and re.search(r"\b(?:e\.g|i\.e|etc|vs|cf|approx)\.$", merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _unqualified_in_process_claims(doc: str) -> list[str]:
    """Sentences that quantify over every concrete trainer AND claim in-process.

    A sentence may say what a *local* trainer does; what it may not do is say
    that *concrete trainers* (all of them) run the training in-process, because
    a transport provider does not run it at all.
    """
    quantifiers = ("concrete trainer", "all trainers", "every trainer")
    properties = ("in-process", "shell out", "subprocess")
    offenders = []
    for sentence in _sentences(doc):
        low = sentence.lower()
        if any(q in low for q in quantifiers) and any(p in low for p in properties):
            offenders.append(sentence)
    return offenders


class _BareTrainer(Trainer):
    """Smallest legal Trainer: implements only the abstract surface.

    Deliberately overrides nothing else, so calls fall through to the
    ``Trainer`` base-class defaults under test.
    """

    @property
    def provider_name(self) -> str:
        return "bare"

    def validate(self, spec: TrainSpec) -> list[str]:
        return []

    def train(self, spec: TrainSpec) -> TrainResult:
        return TrainResult(status="success", job_id="bare-job")


def test_default_status_reports_polling_unsupported() -> None:
    """The default ``status`` returns an actionable error, not a fake verdict.

    A backend that cannot poll a detached job inherits this: the result is a
    terminal ``error`` (never a misleading ``running``/``success``), echoes the
    queried ``job_id``, and names the provider so the caller knows which
    backend declined.
    """
    result = _BareTrainer().status("job-123")

    assert result.status == "error"
    assert result.job_id == "job-123"
    assert "bare" in result.message
    assert "not supported" in result.message


def test_default_latest_checkpoint_returns_none() -> None:
    """A backend with no checkpoint-discovery layout inherits ``None``.

    ``None`` (not an empty string or a raised error) is the documented
    "no discoverable checkpoint" signal that the ``export`` action and resume
    logic branch on.
    """
    assert _BareTrainer().latest_checkpoint("/no/such/output/dir") is None


def test_default_export_returns_checkpoint_dir_unchanged() -> None:
    """HF-native backends need no conversion; the default is a passthrough."""
    trainer = _BareTrainer()
    assert trainer.export(TrainSpec(), "/tmp/ckpt/step_100") == "/tmp/ckpt/step_100"


def test_default_hardware_floor_is_single_24gb_gpu() -> None:
    """The advisory floor defaults to one 24 GB single-node GPU."""
    floor = _BareTrainer().hardware_floor

    assert floor == {"min_gpus": 1, "min_vram_gb": 24, "multinode": False}


# --- The documented train/status contract vs what the providers do ------------


def test_the_measurement_spans_both_train_shapes() -> None:
    """Premise: the providers really do differ in what ``train`` hands back.

    A clean grading result below would prove nothing if every provider were
    terminal-only, so this pins that the scan sees both shapes: several
    providers whose ``train`` is terminal-only, and at least one whose ``train``
    can hand back a job that outlives the call.
    """
    per_provider = {name: _train_statuses(cls) for name, cls in _provider_classes().items()}

    assert len(per_provider) >= 5, f"expected the shipped training providers to be found, got {per_provider}"
    for name, statuses in per_provider.items():
        assert statuses & _TERMINAL_STATUSES, f"{name}: train() reaches no terminal status at all ({statuses})"

    non_terminal = _non_terminal_train_statuses()
    assert non_terminal, (
        "premise: no provider's train() can return a non-terminal status, so the "
        "grading below is vacuous - re-read the walk in _train_statuses"
    )
    terminal_only = sorted(set(per_provider) - set(non_terminal))
    assert terminal_only, "premise: every provider is non-terminal, so nothing pins the local shape"


def test_a_caller_registered_trainer_does_not_enter_the_grading(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trainer registered at runtime must not change what the ABC is graded on.

    The registry is open: a caller (and a sibling test) may register its own
    trainer, whose class lives outside this package and whose ``train`` this
    scan cannot read. Grading it would make the verdict depend on which other
    tests ran first, so the scan is scoped to the providers the package ships.
    """
    shipped = _provider_classes()

    class _CallerTrainer(Trainer):
        @property
        def provider_name(self) -> str:
            return "caller-registered"

        def validate(self, spec: TrainSpec) -> list[str]:
            return []

        def train(self, spec: TrainSpec) -> TrainResult:
            return TrainResult(status="success", job_id="caller-job")

    monkeypatch.setitem(factory._runtime_registry, "caller_registered_probe", lambda: _CallerTrainer)

    assert "caller_registered_probe" in list_trainers(), "premise: the probe did not register"
    assert create_trainer("caller_registered_probe").provider_name == "caller-registered"
    assert _provider_classes() == shipped


def test_the_documented_train_return_admits_every_status_a_provider_returns() -> None:
    """``train``'s prose must name every status a registered ``train`` can return.

    This is the contract a caller branches on. A provider that submits a run
    outliving the process hands back a non-terminal result with a ``job_id`` and
    no ``checkpoint_dir``; prose that describes the return as always terminal
    tells the caller to read "not error" as finished, which renders a
    completed-run report naming an artifact that does not exist yet.
    """
    doc = inspect.getdoc(Trainer.train) or ""
    non_terminal = _non_terminal_train_statuses()

    missing = {
        provider: sorted(status for status in statuses if not _names_the_status_value(doc, status))
        for provider, statuses in non_terminal.items()
    }
    missing = {provider: statuses for provider, statuses in missing.items() if statuses}
    assert not missing, (
        f"Trainer.train's docstring does not name the status(es) these providers' "
        f"train() can return: {missing}. A caller branching on the documented "
        f"contract cannot handle a result the ABC never mentions."
    )


def test_the_documented_status_polling_covers_a_job_its_own_train_submitted() -> None:
    """``status``'s prose must cover the job its own ``train`` hands back.

    Polling is not only for a run started out of band: a transport ``train``
    returns a non-terminal result whose id is polled through this very method,
    so the prose has to name that case or the one route that produces a pollable
    job reads as unsupported.
    """
    doc = inspect.getdoc(Trainer.status) or ""
    non_terminal = sorted({status for statuses in _non_terminal_train_statuses().values() for status in statuses})

    unnamed = [status for status in non_terminal if not _names_the_status_value(doc, status)]
    assert not unnamed, (
        f"Trainer.status's docstring never mentions the {unnamed} result a provider's "
        f"train() hands back, so the job it is the only way to poll is undocumented here."
    )


def test_the_class_docstring_does_not_claim_every_trainer_runs_training_in_process() -> None:
    """No sentence may quantify the in-process claim over all concrete trainers.

    Saying what a *local* trainer does is accurate; saying that *concrete
    trainers* run the backend's training in-process is not, because a transport
    provider imports no training library and runs nothing locally.
    """
    assert _non_terminal_train_statuses(), "premise: no transport-shaped provider is registered"

    offenders = _unqualified_in_process_claims(inspect.getdoc(Trainer) or "")
    assert not offenders, (
        "Trainer's docstring quantifies the in-process execution model over every "
        f"concrete trainer, which a transport provider does not follow: {offenders}"
    )


def test_a_planted_unqualified_claim_is_reported() -> None:
    """The claim scan is load-bearing, not vacuously clean.

    Feeds it prose of both shapes: a sentence that quantifies over concrete
    trainers AND asserts in-process execution must be reported, while a sentence
    that scopes the same property to the local shape must not be.
    """
    planted = "Concrete trainers call the backend's training function in-process."
    scoped = "A local trainer calls the backend's training function in-process."

    assert _unqualified_in_process_claims(planted) == [planted]
    assert _unqualified_in_process_claims(scoped) == []
    assert _unqualified_in_process_claims("Concrete trainers read the fields they support.") == []


def test_naming_a_status_means_naming_the_value_not_the_english_word() -> None:
    """Prose that merely contains the word does not document the return value.

    "a still-running job" says nothing about what ``train`` hands back, so the
    grader keys on the ``literal`` form this package writes a status value in.
    Without this distinction the grading passes on prose that never mentions the
    result at all.
    """
    assert _names_the_status_value("returns ``running`` with a ``job_id``", "running")
    assert not _names_the_status_value("report on a separately launched, still-running job", "running")
    assert not _names_the_status_value("the run is running", "running")


@pytest.mark.parametrize(
    "provider",
    [name for name, cls in _provider_classes().items() if _defining_class(cls, "status") is Trainer],
)
def test_the_default_status_is_the_live_contract_for_every_provider_that_inherits_it(provider: str) -> None:
    """A provider that does not override ``status`` declines by name, not by fiction.

    The default is not a hypothetical inherited by some future backend - it is
    what most shipped providers actually answer, so each of them is driven
    through it here: a terminal ``error`` (never a misleading ``running`` or
    ``success``), echoing the queried id and naming the provider that declined.
    """
    trainer = create_trainer(provider)
    result = trainer.status("job-abc")

    assert result.status == "error"
    assert result.job_id == "job-abc"
    assert trainer.provider_name in result.message
    assert "not supported" in result.message
