"""Every command the prose invokes is one ``python -m strands_robots`` dispatches.

``strands_robots.__main__`` routes its first argv token through a closed list
(``_COMMANDS``) and exits 1 on anything else, so a page naming a command that
list does not carry documents a first step that cannot work. Nothing graded the
two against each other: the dispatcher has its own tests and the pages have
theirs, and neither side reads the other.

Measured on the tree this arrived in, six ``docs/dashboard/*.md`` pages launched
an operator dashboard with ``python -m strands_robots dashboard --port 8090
--local-dev`` on nine lines - a command the dispatcher never carried, with flags
that appear nowhere in the package. One of those lines said its flag table came
from ``--help`` on that build. The pages went; this grader is what keeps the
next such line from landing.

Scope is invocations, not the distribution name: ``strands-robots speaks ROS 2``
is prose about the project and names no command, so only text inside a fenced
block or an inline code span is read, and the invocation has to start it (after
any ``$``, ``sudo`` or ``VAR=value`` prefix a shell line carries). A leading
``-`` token is an option rather than a command - ``--help`` and ``--version``
are answered by the dispatcher directly - and a token that is not a command name
by shape (``strands-robots >= 0.5.1``, a version constraint) names nothing to
dispatch.

The rule is one-directional: a page need not name every command, only nothing
outside the list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from strands_robots.__main__ import _COMMANDS

_REPO_ROOT = Path(__file__).resolve().parent.parent

_FENCE = re.compile(r"^\s*(?:```|~~~)")
_INLINE = re.compile(r"`([^`\n]+)`")
#: Shell noise a documented command line may carry before the command itself.
_PREFIX = re.compile(r"^(?:\$\s*|sudo\s+|[A-Z_][A-Z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S*)\s+)+")
_INVOKE = re.compile(r"^(?:python3?\s+-m\s+strands_robots|strands-robots)\s+([a-z][a-z0-9-]*)\b")


@dataclass(frozen=True)
class _Invocation:
    """A command a page tells the reader to run."""

    page: str
    command: str
    line: str

    def __str__(self) -> str:
        return f"{self.page}: {self.line.strip()} -> {self.command!r}"


def _invocations_in(text: str, page: str) -> list[_Invocation]:
    """The commands ``text`` invokes, read from its code contexts only."""
    found: list[_Invocation] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        spans = [line] if in_fence else [m.group(1) for m in _INLINE.finditer(line)]
        for span in spans:
            match = _INVOKE.match(_PREFIX.sub("", span.strip()))
            if match is not None:
                found.append(_Invocation(page, match.group(1), line))
    return found


def _documented_invocations() -> list[_Invocation]:
    """Every command the operator-facing prose invokes."""
    pages = sorted(_REPO_ROOT.glob("docs/**/*.md")) + [_REPO_ROOT / "README.md"]
    found: list[_Invocation] = []
    for page in pages:
        rel = page.relative_to(_REPO_ROOT).as_posix()
        found += _invocations_in(page.read_text(encoding="utf-8"), rel)
    return found


class TestEveryDocumentedCommandIsDispatched:
    """The command vocabulary the prose invokes is the one the module answers."""

    def test_no_page_invokes_a_command_the_dispatcher_lacks(self) -> None:
        """A documented command must reach a dispatch branch."""
        offenders = [i for i in _documented_invocations() if i.command not in _COMMANDS]
        assert not offenders, "documentation invokes commands python -m strands_robots does not carry:\n" + "\n".join(
            f"  {o}  (dispatched: {', '.join(_COMMANDS)})" for o in offenders
        )

    def test_every_shipped_command_is_documented_somewhere(self) -> None:
        """The other direction, and the proof the sweep read anything.

        A grader that finds nothing reports a clean tree, so the population is
        pinned to the dispatcher's own list rather than to a count: each command
        has to be invoked by some page, and more than one page has to be
        reached. ``doctor`` was documented only in the deleted pages, so this is
        also what keeps a shipped command from losing its last invocation.
        """
        found = _documented_invocations()
        assert {i.command for i in found} == set(_COMMANDS), f"read {sorted({i.command for i in found})}"
        assert len({i.page for i in found}) >= 2, "the sweep must reach more than one page"


class TestTheGraderIsLoadBearing:
    """The sweep reports a planted invocation, and only an undispatched one."""

    def test_a_planted_unknown_command_is_reported(self) -> None:
        """The shape the deleted pages used must fail on arrival."""
        planted = _invocations_in("```bash\npython -m strands_robots cockpit --port 8090\n```", "planted.md")
        assert [i.command for i in planted] == ["cockpit"]
        assert [i for i in planted if i.command not in _COMMANDS] == planted

    def test_a_planted_dispatched_command_is_accepted(self) -> None:
        """Both spellings of a real command must pass."""
        planted = _invocations_in("`strands-robots doctor` and `python -m strands_robots verify-dataset ds`", "p.md")
        assert [i.command for i in planted] == ["doctor", "verify-dataset"]
        assert [i for i in planted if i.command not in _COMMANDS] == []

    def test_prose_naming_the_project_is_not_an_invocation(self) -> None:
        """The distribution name in a sentence invokes nothing."""
        assert _invocations_in("strands-robots speaks ROS 2 from four angles.", "p.md") == []
        assert _invocations_in("Needs `strands-robots >= 0.5.1`, so upgrade.", "p.md") == []

    def test_an_option_is_not_a_command(self) -> None:
        """``--help`` and ``--version`` are answered without a command."""
        assert _invocations_in("```bash\nstrands-robots --help\n```", "p.md") == []
