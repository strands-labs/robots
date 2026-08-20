"""A documented trainer constructor argument has to be one the constructor declares.

Every in-package :class:`~strands_robots.training.base.Trainer` takes ``**kwargs``,
so an undeclared keyword is *absorbed* rather than refused. That makes a prose claim
about a constructor argument unfalsifiable at the call site: the caller passes it, the
constructor accepts it, and the value is dropped without a log line. The docs then also
describe the execution model the argument would have selected, so a stale claim comes in
pairs - a knob that does nothing, plus install guidance built on top of it.

The assertions here grade the prose against the code:

* an ``accept ... `name=` ... argument`` claim about a trainer class must name an
  *explicit* parameter of that class, and ``**kwargs`` absorption does not count;
* a provider whose backend is imported in the calling interpreter (measured: its module
  references neither ``sys.executable`` nor ``subprocess``, so it has no interpreter to
  select) must have its dependency row say so, because that is what decides which
  environment an operator installs the backend into.

Both are derived from the trainer registry and from ``inspect`` rather than from a table
listed here, so a provider added later is graded on arrival.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from strands_robots.training.base import Trainer
from strands_robots.training.factory import import_trainer_class, list_trainers

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs"
_TRAINING_OVERVIEW = _DOCS / "training" / "overview.md"

# Providers the dependency table has to speak about. Their backends are third-party
# checkouts, so *which interpreter imports them* is the operator-visible question.
_THIRD_PARTY_BACKENDS = ("groot", "cosmos3")

# An acceptance claim: a verb of acceptance and a `name=` token in one sentence.
_ACCEPTS_ARGUMENT = re.compile(
    r"(?:accepts?|takes?|supports?)\b[^.]{0,200}?`(?P<arg>[a-z_][a-z0-9_]*)=`[^.]{0,80}?\bargument",
    re.IGNORECASE | re.DOTALL,
)

# The dependency table, addressed by its own header so the capability table further
# up the page - whose "Launcher" column describes the *upstream* project, not this
# one - is not graded as if it were install guidance.
_DEPENDENCY_TABLE_HEADER = "| Provider / policy | Install | Notes |"


def _doc_pages() -> list[Path]:
    """Every user-facing markdown page, plus the README."""
    pages = sorted(_DOCS.rglob("*.md"))
    readme = _REPO_ROOT / "README.md"
    if readme.exists():
        pages.append(readme)
    return pages


def _trainer_classes() -> dict[str, type[Trainer]]:
    """Registered trainer classes this package defines, keyed by provider name.

    A provider whose class lives outside ``strands_robots`` (a runtime registration
    from a caller) is skipped: its constructor is not this package's contract.
    """
    resolved: dict[str, type[Trainer]] = {}
    for provider in sorted(list_trainers()):
        try:
            cls = import_trainer_class(provider)
        except Exception:  # noqa: BLE001 - an unresolvable provider is not this contract
            continue
        if cls.__module__.startswith("strands_robots."):
            resolved[provider] = cls
    return resolved


def _explicit_parameters(cls: type) -> set[str]:
    """The parameter names *cls* declares, excluding the variadic sinks.

    ``**kwargs`` is deliberately excluded: absorbing a keyword is not accepting it,
    and treating it as acceptance is what let a documented argument that no
    constructor declares read as supported.
    """
    variadic = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    return {p.name for p in inspect.signature(cls).parameters.values() if p.kind not in variadic}


def _absorbs_unknown_keywords(cls: type) -> bool:
    """Whether *cls* has a ``**kwargs`` sink that swallows an undeclared keyword."""
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(cls).parameters.values())


def _selects_an_interpreter(cls: type) -> list[str]:
    """Interpreter-spawning references in the module that defines *cls*.

    A trainer that never reads ``sys.executable`` and never calls into
    ``subprocess`` has no second interpreter to run its backend in, so its backend
    has to be importable from the one that imports ``strands_robots``.
    """
    source = Path(inspect.getsourcefile(cls) or "").read_text(encoding="utf-8")
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and ast.unparse(node) == "sys.executable":
            found.add("sys.executable")
        elif isinstance(node, ast.Call):
            call = ast.unparse(node.func)
            if call.startswith("subprocess.") or call.endswith(".Popen"):
                found.add(call)
    return sorted(found)


def _dependency_table_row(provider: str) -> str:
    """The install/notes row for *provider*, from the dependency table only.

    The page carries an earlier capability table keyed on the same provider names;
    its columns describe each upstream project rather than how this package drives
    it, so only rows under the dependency-table header are install guidance.
    """
    lines = _TRAINING_OVERVIEW.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(_DEPENDENCY_TABLE_HEADER)
    except ValueError:  # pragma: no cover - guarded by the non-vacuity test
        return ""
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        if line.startswith(f"| `{provider}` |"):
            return line
    return ""


def _page_label(page: Path) -> str:
    """A repository-relative label for *page*, falling back to its absolute path."""
    try:
        return str(page.relative_to(_REPO_ROOT))
    except ValueError:
        return str(page)


def _paragraph_around(text: str, index: int) -> str:
    """The blank-line-delimited block of *text* containing *index*.

    A multi-line blockquote is one such block, which keeps the subject of a claim
    and the claim itself together however the prose is wrapped.
    """
    start = text.rfind("\n\n", 0, index)
    start = 0 if start < 0 else start + 2
    end = text.find("\n\n", index)
    end = len(text) if end < 0 else end
    return text[start:end]


def _unbacked_argument_claims(pages: list[Path]) -> dict[str, list[str]]:
    """Documented ``name=`` arguments that no named trainer class declares."""
    classes = _trainer_classes()
    by_class_name = {cls.__name__: cls for cls in classes.values()}
    offenders: dict[str, list[str]] = {}
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for match in _ACCEPTS_ARGUMENT.finditer(text):
            arg = match.group("arg")
            # The subject usually precedes the verb ("A / B / C accept a `x=` argument"),
            # so resolve names over the whole paragraph rather than the matched span.
            named = [name for name in by_class_name if name in _paragraph_around(text, match.start())]
            if not named:
                continue
            undeclared = [name for name in named if arg not in _explicit_parameters(by_class_name[name])]
            if undeclared:
                offenders.setdefault(_page_label(page), []).append(f"{arg}= on {', '.join(sorted(undeclared))}")
    return offenders


# --- the claim has to match the constructor ---


def test_no_document_claims_a_trainer_argument_the_constructor_does_not_declare() -> None:
    """A documented ``name=`` argument is a real parameter of the class it names."""
    offenders = _unbacked_argument_claims(_doc_pages())
    assert not offenders, (
        "documentation promises trainer constructor arguments that are not declared "
        f"(each is absorbed by **kwargs and dropped): {offenders}"
    )


@pytest.mark.parametrize("provider", _THIRD_PARTY_BACKENDS)
def test_absorbing_a_keyword_is_not_accepting_it(provider: str) -> None:
    """An undeclared keyword is taken and dropped, so prose cannot lean on ``**kwargs``.

    This is the premise behind excluding ``**kwargs`` from the declared set: the
    constructor call a reader would make succeeds and leaves nothing behind.
    """
    cls = _trainer_classes()[provider]
    assert _absorbs_unknown_keywords(cls), f"premise: {cls.__name__} no longer has a **kwargs sink"
    sentinel = "not_a_declared_trainer_argument"
    assert sentinel not in _explicit_parameters(cls)
    instance = cls(**{sentinel: "/some/venv/bin/python"})
    assert not hasattr(instance, sentinel), (
        f"{cls.__name__} now records {sentinel}; the absorption premise needs re-checking"
    )


# --- the execution model the install guidance rests on ---


@pytest.mark.parametrize("provider", _THIRD_PARTY_BACKENDS)
def test_no_in_package_trainer_can_select_an_interpreter(provider: str) -> None:
    """No trainer spawns an interpreter, so none can run its backend in another one."""
    cls = _trainer_classes()[provider]
    spawns = _selects_an_interpreter(cls)
    assert not spawns, (
        f"{cls.__name__} now reaches for {spawns}; if a trainer can select an interpreter "
        "again the dependency table's same-environment guidance needs revisiting"
    )


@pytest.mark.parametrize("provider", _THIRD_PARTY_BACKENDS)
def test_the_dependency_row_states_which_interpreter_imports_the_backend(provider: str) -> None:
    """The row tells an operator the backend is imported in the calling interpreter.

    Stated positively on purpose: the row has to carry the requirement, so it cannot
    be satisfied by dropping a phrase, and a correctly qualified mention of the
    alternative does not trip it.
    """
    row = _dependency_table_row(provider)
    assert row, f"no dependency-table row for `{provider}` in {_TRAINING_OVERVIEW.name}"
    lowered = row.lower()
    assert "calling interpreter" in lowered or "same environment" in lowered.replace("**", ""), (
        f"the `{provider}` dependency row does not say which interpreter imports the backend, "
        f"so an operator cannot tell which environment to install it into: {row}"
    )


def test_the_runtime_refusal_asks_for_the_same_interpreter(caplog: pytest.LogCaptureFixture) -> None:
    """The failure an operator actually hits names the calling interpreter.

    This is the message the dependency guidance has to agree with; if the runtime
    ever starts accepting a separately installed backend, this is the assertion that
    should be revisited first.
    """
    del caplog
    for provider in _THIRD_PARTY_BACKENDS:
        module = inspect.getmodule(_trainer_classes()[provider])
        hint = getattr(module, "_INSTALL_HINT", "")
        assert "this interpreter" in hint, f"{provider} install hint no longer names the interpreter: {hint!r}"
        assert "same" in hint, f"{provider} install hint no longer asks for the same environment: {hint!r}"


# --- non-vacuity ---


def test_the_sweep_reaches_the_training_documentation() -> None:
    """The graded corpus really contains the page that carries this guidance."""
    pages = _doc_pages()
    assert _TRAINING_OVERVIEW in pages
    assert len(pages) >= 20, f"only {len(pages)} documentation pages discovered"
    assert set(_THIRD_PARTY_BACKENDS) <= set(_trainer_classes()), (
        f"graded providers missing from the registry: {sorted(_trainer_classes())}"
    )


def test_a_planted_claim_is_reported_and_a_denial_is_not(tmp_path: Path) -> None:
    """An acceptance claim is graded; saying the argument does not exist is not a claim."""
    claim = tmp_path / "claim.md"
    claim.write_text(
        "`Gr00tTrainer` accepts a `python_executable=` argument (defaults to `sys.executable`).\n",
        encoding="utf-8",
    )
    reported = _unbacked_argument_claims([claim])
    assert reported, "a planted claim about an undeclared argument was not reported"
    assert "python_executable= on Gr00tTrainer" in next(iter(reported.values()))

    denial = tmp_path / "denial.md"
    denial.write_text(
        "`Gr00tTrainer` calls its backend in the same interpreter. There is no `python_executable=` argument to set.\n",
        encoding="utf-8",
    )
    assert not _unbacked_argument_claims([denial]), "a denial was mistaken for a claim"

    real = tmp_path / "real.md"
    real.write_text("`Gr00tTrainer` accepts a `groot_root=` argument.\n", encoding="utf-8")
    assert not _unbacked_argument_claims([real]), "a declared argument was reported"
