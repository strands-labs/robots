"""Documentation must state the mesh opt-in the factory implements.

Joining the Zenoh mesh is opt-in. ``Robot()`` and ``Simulation()`` both default
``mesh`` to ``None``, which consults ``STRANDS_MESH``; unset leaves the mesh off,
so ``robot.mesh`` is ``None`` and every attribute read on it raises
:class:`AttributeError`. That has been the behaviour since the quiet-by-default
change, but three pages still describe the world before it.

A page can get this wrong in three independent spellings, and each fails
differently:

* A **runnable snippet** that constructs a robot without opting in and then
  reads ``robot.mesh.peers`` does not run slowly or warn - copied as written it
  raises ``AttributeError: 'NoneType' object has no attribute 'peers'`` on that
  line.
* **Prose** that says every ``Robot()`` auto-joins tells the reader they are in a
  state they are not in, and it hides the only spelling that would put them
  there.
* A **parameter table** whose ``mesh`` Default cell says ``True`` compounds both:
  the reader's documented lever (``mesh=False``) is a no-op for the state they
  are actually in, while the lever that reaches the documented state is never
  shown.

Two sibling guards grade the same contract at narrower scopes:
``tests/mesh/test_mesh_env_opt_in_documented_default.py`` grades a table whose
first cell is the environment variable, and
``tests/test_docs_robot_factory_reference.py`` grades the factory reference
page's own parameter table and Mesh section. Neither reads any other page, and
neither grades a snippet. This module grades every ``docs/**/*.md`` page and
``README.md``, at the scope each spelling lives in: per fenced block for the
snippets, per paragraph for the prose, per row for the tables.

The snippet check is deliberately block-scoped rather than page-scoped: a reader
copies one fence, so the opt-in has to be visible inside the fence they copy.
Constructions the block does not perform are not graded - a variable that
arrives from elsewhere degrades to "not graded", never to a failure.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import strands_robots
from strands_robots.robot import Robot, _mesh_env_opt_in

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent

_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

# A construction whose result is bound to a name, e.g. ``sim_a = Robot("so100")``.
_CONSTRUCTION = re.compile(r"^\s*(\w+)\s*=\s*(Robot|Simulation)\s*\(", re.MULTILINE)

# ``STRANDS_MESH`` set inside the block, in either an ``os.environ[...] = "..."``
# assignment or a shell-style ``STRANDS_MESH=...`` comment.
_ENV_SET = re.compile(r"STRANDS_MESH(?![A-Z_])[\"']?\]?\s*[=:]\s*[\"']?([A-Za-z01]+)")

# ``Robot`` takes the boolean opt-in. A ``Simulation`` built directly does not:
# its ``mesh=`` takes an already-started client, and ``Simulation(mesh=True)``
# raises TypeError, so the opt-in there is assigning a started client. Grading
# both the same way would let this guard bless a snippet that cannot construct.
_OPTS_IN_BY_ARGUMENT = re.compile(r"\bmesh\s*=\s*True\b")
_ASSIGNS_A_CLIENT = "{name}.mesh\\s*=\\s*(?!None\\b)"

# Prose that claims the mesh is joined without asking. Both halves must appear in
# one paragraph: a bare construction, and a claim that it happens by itself.
_BARE_CONSTRUCTION = re.compile(r"`Robot\(\)`|`Simulation\(\)`|Every\s+`Robot\(\)", re.IGNORECASE)
_AUTOMATIC_CLAIM = re.compile(r"auto-?join\w*|automatical\w*|no setup|out of the box", re.IGNORECASE)

# A paragraph that names the spelling which joins is stating the contract rather
# than the old default, so a page stays free to write "a bare ``Robot()`` does
# not auto-join - pass ``mesh=True``". The exemption is deliberately the spelling
# and not a reassuring word: ``STRANDS_MESH_MULTICAST`` is described as opt-in in
# the same paragraph that used to claim joining was automatic.
_NAMES_THE_OPT_IN = _OPTS_IN_BY_ARGUMENT

# A markdown table row split into cells on unescaped pipes only, so a Type cell
# that spells a union as ``bool \| None`` keeps the row intact.
_CELL_SPLIT = re.compile(r"(?<!\\)\|")

# The guard is only meaningful while it still reaches the pages. Well under the
# real counts, so a page added or removed does not churn these, but far enough
# above zero that a reformat which stops matching fails loudly instead of
# reporting clean.
_MINIMUM_PAGES = 30
_MINIMUM_FENCES = 150
_MINIMUM_PARAGRAPHS = 800

_PROBE_MJCF = """<mujoco model="probe">
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 0.1"/>
    <body name="link0" pos="0 0 0.1">
      <joint name="joint0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0  0 0 0.2"/>
    </body>
  </worldbody>
  <actuator><motor joint="joint0" ctrlrange="-1 1"/></actuator>
</mujoco>"""


def _pages() -> list[Path]:
    """Return every documentation page this guard grades."""
    return sorted(_REPO_ROOT.glob("docs/**/*.md")) + [_REPO_ROOT / "README.md"]


def _enabling_spellings(text: str) -> list[str]:
    """Return the ``STRANDS_MESH`` values in ``text`` that really opt in.

    Args:
        text: A fenced block or a page, searched for ``STRANDS_MESH`` settings.

    Returns:
        The matched values for which :func:`_mesh_env_opt_in` reports ``True``,
        graded by calling it rather than by comparing against a copied list of
        accepted spellings.
    """
    import os

    found: list[str] = []
    previous = os.environ.get("STRANDS_MESH")
    try:
        for match in _ENV_SET.finditer(text):
            os.environ["STRANDS_MESH"] = match.group(1)
            if _mesh_env_opt_in():
                found.append(match.group(1))
    finally:
        if previous is None:
            os.environ.pop("STRANDS_MESH", None)
        else:
            os.environ["STRANDS_MESH"] = previous
    return found


def _call_text(block: str, start: int) -> str:
    """Return the construction call beginning at ``start``, parentheses balanced.

    Args:
        block: The fenced block's source text.
        start: Index of the first character of the assignment.

    Returns:
        The substring through the call's closing parenthesis, or the remainder of
        the block when the parentheses never balance.
    """
    # ``^\s*`` can begin the match on a preceding blank line, so trim to the
    # first real character or the report names an empty string.
    start += len(block[start:]) - len(block[start:].lstrip())
    depth = 0
    for index in range(block.index("(", start), len(block)):
        if block[index] == "(":
            depth += 1
        elif block[index] == ")":
            depth -= 1
            if depth == 0:
                return block[start : index + 1]
    return block[start:]


def _mesh_reads_without_opt_in(text: str, page: str) -> list[str]:
    """Return one report per ``.mesh`` read on a robot the block never opted in.

    Args:
        text: A page's full markdown source.
        page: The page's repository-relative path, used in the reports.

    Returns:
        A list of human-readable offender descriptions, each naming the page, the
        line of the read, and the construction that omitted the opt-in.
    """
    reports: list[str] = []
    for fence in _PYTHON_FENCE.finditer(text):
        block = fence.group(1)
        block_first_line = text[: fence.start()].count("\n") + 2
        constructions = {
            match.group(1): (match.group(2), _call_text(block, match.start()))
            for match in _CONSTRUCTION.finditer(block)
        }
        if not constructions:
            continue
        env_opt_in = bool(_enabling_spellings(block))
        for name, (kind, call) in constructions.items():
            if kind == "Simulation":
                if re.search(_ASSIGNS_A_CLIENT.format(name=re.escape(name)), block):
                    continue
            elif env_opt_in or _OPTS_IN_BY_ARGUMENT.search(call):
                continue
            for read in re.finditer(rf"\b{re.escape(name)}\.mesh\.(\w+)", block):
                lineno = block_first_line + block[: read.start()].count("\n")
                reports.append(
                    f"{page}:{lineno}: reads `{name}.mesh.{read.group(1)}` but "
                    f"`{call.splitlines()[0].strip()}` never opts in"
                )
    return reports


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, text)`` for each prose paragraph outside a fence.

    Args:
        text: A page's full markdown source.

    Returns:
        One entry per blank-line-delimited paragraph, with its lines joined by a
        single space so a claim spanning a line break is still one string.
    """
    out: list[tuple[int, str]] = []
    inside_fence = False
    buffer: list[str] = []
    start = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            inside_fence = not inside_fence
            if buffer:
                out.append((start, " ".join(buffer)))
                buffer = []
            continue
        if inside_fence:
            continue
        if not line.strip():
            if buffer:
                out.append((start, " ".join(buffer)))
                buffer = []
            continue
        if not buffer:
            start = lineno
        buffer.append(line.strip())
    if buffer:
        out.append((start, " ".join(buffer)))
    return out


def _automatic_join_claims(text: str, page: str) -> list[str]:
    """Return one report per paragraph claiming a bare construction joins."""
    reports = []
    for lineno, paragraph in _paragraphs(text):
        if "mesh" not in paragraph.lower():
            continue
        if not (_AUTOMATIC_CLAIM.search(paragraph) and _BARE_CONSTRUCTION.search(paragraph)):
            continue
        if _NAMES_THE_OPT_IN.search(paragraph) or _enabling_spellings(paragraph):
            continue
        reports.append(f"{page}:{lineno}: {paragraph[:140]}")
    return reports


def _mesh_default_cells(text: str, page: str) -> list[tuple[str, str]]:
    """Return ``(report_prefix, default_cell)`` for each ``mesh`` parameter row.

    Args:
        text: A page's full markdown source.
        page: The page's repository-relative path.

    Returns:
        One entry per four-cell table row whose first cell is the backticked
        parameter name ``mesh``. Rows in other shapes - notably a two-column knob
        table, or a table keyed on the environment variable - are left to the
        sibling guards named in this module's docstring.
    """
    found = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in _CELL_SPLIT.split(stripped.strip("|"))]
        if len(cells) != 4 or cells[0] != "`mesh`":
            continue
        found.append((f"{page}:{lineno}", cells[2]))
    return found


class TestTheScanReachesThePages:
    """Premises. A clean result must mean the pages were read, not skipped."""

    def test_the_pages_are_found(self) -> None:
        pages = _pages()
        assert (_REPO_ROOT / "README.md").is_file(), "premise: README.md is one of the graded pages"
        assert len(pages) >= _MINIMUM_PAGES, (
            f"premise: only {len(pages)} documentation page(s) found, below the "
            f"{_MINIMUM_PAGES} this guard expects. A clean run would prove nothing."
        )

    def test_enough_python_fences_are_scanned(self) -> None:
        fences = sum(len(_PYTHON_FENCE.findall(page.read_text(encoding="utf-8"))) for page in _pages())
        assert fences >= _MINIMUM_FENCES, (
            f"premise: only {fences} python fence(s) matched, below the {_MINIMUM_FENCES} "
            "this guard expects. A fence spelling change would silently empty the scan."
        )

    def test_enough_paragraphs_are_scanned(self) -> None:
        paragraphs = sum(len(_paragraphs(page.read_text(encoding="utf-8"))) for page in _pages())
        assert paragraphs >= _MINIMUM_PARAGRAPHS, (
            f"premise: only {paragraphs} prose paragraph(s) matched, below the "
            f"{_MINIMUM_PARAGRAPHS} this guard expects."
        )

    def test_some_snippet_reads_the_mesh_attribute(self) -> None:
        """The subject has to exist: some page must actually use ``.mesh``."""
        reads = 0
        for page in _pages():
            for fence in _PYTHON_FENCE.finditer(page.read_text(encoding="utf-8")):
                reads += len(re.findall(r"\w+\.mesh\.\w+", fence.group(1)))
        assert reads > 0, (
            "premise: no documented snippet reads a `.mesh` attribute, so the "
            "opt-in this guard grades would never be needed."
        )


class TestEveryRunnableSnippetOptsIn:
    """A fence a reader copies must reach the mesh it goes on to use."""

    def test_no_snippet_reads_mesh_without_opting_in(self) -> None:
        offenders: list[str] = []
        for page in _pages():
            rel = page.relative_to(_REPO_ROOT).as_posix()
            offenders += _mesh_reads_without_opt_in(page.read_text(encoding="utf-8"), rel)
        assert not offenders, (
            "Joining the mesh is opt-in, so these snippets raise AttributeError on "
            "None where they read a `.mesh` attribute. Pass `mesh=True` to the "
            "construction, or set an enabling `STRANDS_MESH` in the same block:\n  " + "\n  ".join(offenders)
        )


class TestNoPageClaimsABareRobotJoins:
    """Prose must not put the reader in a state a bare construction skips."""

    def test_no_paragraph_claims_an_automatic_join(self) -> None:
        offenders: list[str] = []
        for page in _pages():
            rel = page.relative_to(_REPO_ROOT).as_posix()
            offenders += _automatic_join_claims(page.read_text(encoding="utf-8"), rel)
        assert not offenders, (
            "A bare `Robot()` or `Simulation()` does not join the mesh - `mesh` "
            "defaults to None and `STRANDS_MESH` is unset - so these paragraphs "
            "describe a state the reader is not in and omit the spelling that "
            "would reach it:\n  " + "\n  ".join(offenders)
        )


class TestAMeshParameterRowStatesTheRealDefault:
    """A Default cell states the value a caller gets by omitting the parameter."""

    def test_no_mesh_row_documents_a_default_that_joins(self) -> None:
        real_default = inspect.signature(Robot).parameters["mesh"].default
        offenders: list[str] = []
        for page in _pages():
            rel = page.relative_to(_REPO_ROOT).as_posix()
            for where, cell in _mesh_default_cells(page.read_text(encoding="utf-8"), rel):
                token = cell.strip("`")
                try:
                    documented = ast.literal_eval(token)
                except (SyntaxError, ValueError):
                    continue  # A prose default; the env-var guard grades that spelling.
                if documented != real_default:
                    offenders.append(
                        f"{where}: documents {cell} but omitting `mesh` yields {real_default!r}, "
                        "which leaves the mesh off"
                    )
        assert not offenders, (
            "The `mesh` Default column names a state a bare construction is not in, "
            "and so recommends a lever that changes nothing:\n  " + "\n  ".join(offenders)
        )


class TestTheRealityThePagesDescribe:
    """Controls. These hold before and after the pages are corrected."""

    def test_a_bare_robot_leaves_mesh_off(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        pytest.importorskip("mujoco")
        monkeypatch.delenv("STRANDS_MESH", raising=False)
        mjcf = Path(str(tmp_path)) / "probe.xml"
        mjcf.write_text(_PROBE_MJCF, encoding="utf-8")
        robot = Robot("so100", mode="sim", urdf_path=str(mjcf))
        try:
            assert robot.mesh is None
        finally:
            robot.destroy()

    @pytest.mark.parametrize("spelling", ["true", "1", "yes"])
    def test_the_spellings_the_pages_offer_really_opt_in(self, monkeypatch: pytest.MonkeyPatch, spelling: str) -> None:
        monkeypatch.setenv("STRANDS_MESH", spelling)
        assert _mesh_env_opt_in() is True

    @pytest.mark.parametrize("spelling", ["false", "0", "no", "on", "enabled", ""])
    def test_a_plausible_non_opt_in_spelling_stays_off(self, monkeypatch: pytest.MonkeyPatch, spelling: str) -> None:
        """Only three spellings opt in, so no page may offer a fourth."""
        monkeypatch.setenv("STRANDS_MESH", spelling)
        assert _mesh_env_opt_in() is False


class TestTheGuardWouldCatchARegression:
    """A clean sweep must mean the pages are right, not that nothing is graded."""

    def test_a_planted_snippet_is_reported(self) -> None:
        planted = '```python\nfrom strands_robots import Robot\n\nr = Robot("so100")\nprint(r.mesh.peers)\n```\n'
        assert _mesh_reads_without_opt_in(planted, "planted.md")

    def test_a_planted_snippet_that_opts_in_is_not_reported(self) -> None:
        by_argument = '```python\nr = Robot("so100", mesh=True)\nprint(r.mesh.peers)\n```\n'
        by_env = '```python\nos.environ["STRANDS_MESH"] = "true"\nr = Robot("so100")\nprint(r.mesh.peers)\n```\n'
        assert not _mesh_reads_without_opt_in(by_argument, "planted.md")
        assert not _mesh_reads_without_opt_in(by_env, "planted.md")

    def test_a_planted_simulation_that_assigns_a_client_is_not_reported(self) -> None:
        planted = (
            '```python\nsim = Simulation(tool_name="sim")\n'
            'sim.mesh = init_mesh(sim, peer_id="bench")\nprint(sim.mesh.peers)\n```\n'
        )
        assert not _mesh_reads_without_opt_in(planted, "planted.md")

    def test_a_planted_simulation_passing_true_is_still_reported(self) -> None:
        """``Simulation(mesh=True)`` raises TypeError, so it is not an opt-in."""
        planted = '```python\nsim = Simulation(tool_name="sim", mesh=True)\nprint(sim.mesh.peers)\n```\n'
        assert _mesh_reads_without_opt_in(planted, "planted.md")

    def test_a_planted_paragraph_is_reported(self) -> None:
        planted = "Every `Robot()` auto-joins the mesh.\n"
        assert _automatic_join_claims(planted, "planted.md")

    def test_a_planted_paragraph_naming_the_spelling_is_not_reported(self) -> None:
        """A page may say a bare construction does *not* auto-join."""
        planted = "A bare `Robot()` does not auto-join the mesh - pass `mesh=True`.\n"
        assert not _automatic_join_claims(planted, "planted.md")

    def test_a_planted_paragraph_that_only_says_opt_in_is_still_reported(self) -> None:
        """A reassuring word about another knob must not earn the exemption."""
        planted = (
            "Every `Robot()` automatically joins the mesh. Multicast scouting is "
            "off by default and is opt-in via `STRANDS_MESH_MULTICAST=true`.\n"
        )
        assert _automatic_join_claims(planted, "planted.md")

    def test_a_planted_claim_inside_a_fence_is_not_reported(self) -> None:
        """Prose grading must not read code comments as claims."""
        planted = '```python\n# Every `Robot()` auto-joins the mesh\nr = Robot("so100", mesh=True)\n```\n'
        assert not _automatic_join_claims(planted, "planted.md")

    def test_a_planted_table_row_is_found(self) -> None:
        planted = "| `mesh` | `bool` | `True` | Auto-join the Zenoh mesh |\n"
        assert _mesh_default_cells(planted, "planted.md") == [("planted.md:1", "`True`")]
