"""The factory reference page must not state a contract the factory does not implement.

``docs/getting-started/robot-factory.md`` is the reference a caller reads before
writing ``Robot(...)``: a parameter table with a Default column, and a Mesh
section with a copy-paste snippet. Both are hand-written, so either can outlive
the code.

A wrong Default is worse than a missing one. It names a state the caller is
never in, and it hides the knob that would reach the state the page describes -
the reader's only documented lever turns off something that was never on. A
documented refusal that does not happen fails the same way in reverse: the
caller believes a typo is caught at construction and ships the typo.

These tests grade the page against :func:`strands_robots.robot.Robot` itself -
its signature, its docstring and its observed behaviour - rather than against a
hand-copied expectation, so changing a default cannot silently invalidate them.
``tests/mesh/test_mesh_env_opt_in_documented_default.py`` grades the same
opt-in contract where a table's first cell is the environment variable; this
page states it as a constructor default instead, which that guard does not read.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import re
from pathlib import Path

import pytest

from strands_robots.robot import Robot, _mesh_env_opt_in

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOC = _REPO_ROOT / "docs" / "getting-started" / "robot-factory.md"

_VARIADIC = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)

# The guard is only meaningful while it still reaches the table. If a reformat
# or a rename drops the rows, fail loudly instead of reporting clean.
_MINIMUM_GRADED_ROWS = 8

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


def _rows() -> list[tuple[int, str, str, str]]:
    """Return the parameter table's rows.

    Returns:
        A list of ``(line_number, parameter_name, default_cell, description_cell)``
        tuples for every row whose first cell is a single backticked identifier
        (``**kwargs`` included, its leading asterisks stripped).
    """
    found: list[tuple[int, str, str, str]] = []
    for lineno, line in enumerate(_DOC.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        # A Type cell may legitimately spell a union as ``bool \\| None``, so split
        # on unescaped pipes only - a naive split would drop the row and report
        # the parameter as undocumented.
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped.strip("|"))]
        if len(cells) != 4:
            continue
        match = re.fullmatch(r"`\*{0,2}([A-Za-z_][A-Za-z0-9_]*)`", cells[0])
        if match:
            found.append((lineno, match.group(1), cells[2], cells[3]))
    return found


def _mesh_section() -> str:
    """Return the page's ``## Mesh`` section, up to the next heading."""
    text = _DOC.read_text(encoding="utf-8")
    start = text.index("\n## Mesh")
    rest = text[start + 1 :]
    end = rest.find("\n## ", 1)
    return rest if end == -1 else rest[:end]


def _graded() -> list[tuple[int, str, str, inspect.Parameter]]:
    """Return the rows that name a real, non-variadic parameter."""
    out = []
    for lineno, name, default_cell, _ in _rows():
        param = inspect.signature(Robot).parameters.get(name)
        if param is not None and param.kind not in _VARIADIC:
            out.append((lineno, name, default_cell, param))
    return out


class TestTheTableReachesTheSignature:
    """Premises. A clean result must mean the page was read, not skipped."""

    def test_the_page_ships(self) -> None:
        assert _DOC.is_file(), f"premise: {_DOC.relative_to(_REPO_ROOT)} is the page under test"

    def test_enough_rows_are_graded(self) -> None:
        graded = _graded()
        assert len(graded) >= _MINIMUM_GRADED_ROWS, (
            f"premise: only {len(graded)} row(s) of the parameter table resolved to a "
            f"Robot() parameter, below the {_MINIMUM_GRADED_ROWS} this guard expects. "
            "A clean run would prove nothing."
        )


class TestEveryDocumentedDefaultIsTheRealDefault:
    """A Default cell states the value the caller gets by omitting the parameter."""

    def test_no_row_states_a_default_the_signature_contradicts(self) -> None:
        wrong: list[str] = []
        for lineno, name, cell, param in _graded():
            shown = cell.strip("`")
            if param.default is inspect.Parameter.empty:
                if shown.lower() != "required":
                    wrong.append(f"line {lineno}: `{name}` documents {cell} but has no default")
                continue
            try:
                parsed = ast.literal_eval(shown)
            except (SyntaxError, ValueError):
                wrong.append(f"line {lineno}: `{name}` documents {cell}, which is not a literal value")
                continue
            if parsed != param.default or type(parsed) is not type(param.default):
                wrong.append(f"line {lineno}: `{name}` documents {cell} but omitting it yields {param.default!r}")
        assert not wrong, (
            "The Default column names a value the caller never gets, so the row describes "
            "a state a bare Robot() is not in:\n  " + "\n  ".join(wrong)
        )

    def test_every_parameter_has_a_row(self) -> None:
        documented = {name for _, name, _, _ in _rows()}
        missing = [
            name
            for name, param in inspect.signature(Robot).parameters.items()
            if param.kind not in _VARIADIC and name not in documented
        ]
        assert not missing, f"Robot() accepts {missing}, which the parameter table never lists"


class TestADocumentedRefusalReallyRefuses:
    """A row that promises an exception is graded by raising it, not by wording."""

    def test_a_promised_unknown_kwarg_refusal_happens(self, tmp_path: pytest.TempPathFactory) -> None:
        rows = {name: description for _, name, _, description in _rows()}
        description = rows.get("kwargs", "")
        promised = re.search(r"raises?\s+`([A-Za-z_][A-Za-z0-9_]*)`", description)
        if promised is None:
            return  # The row promises no refusal, so there is nothing to verify.
        pytest.importorskip("mujoco")
        exc = getattr(builtins, promised.group(1), None)
        assert isinstance(exc, type) and issubclass(exc, BaseException), (
            f"the `**kwargs` row promises `{promised.group(1)}`, which is not a builtin exception"
        )
        mjcf = Path(str(tmp_path)) / "probe.xml"
        mjcf.write_text(_PROBE_MJCF, encoding="utf-8")
        sim = None
        try:
            with pytest.raises(exc):
                sim = Robot(
                    "so100",
                    mode="sim",
                    urdf_path=str(mjcf),
                    definitely_not_a_forwardable_kwarg=1,
                )
        finally:
            if sim is not None:
                sim.destroy()


class TestTheMeshSectionNamesTheSpellingThatEnablesMesh:
    """The Mesh section must reach the state its snippet prints."""

    def test_the_section_names_an_enabling_spelling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        section = _mesh_section()
        candidates = {
            *(m.group(1) for m in re.finditer(r"STRANDS_MESH\s*=\s*([A-Za-z01]+)", section)),
        }
        enabling = []
        for raw in candidates:
            monkeypatch.setenv("STRANDS_MESH", raw)
            if _mesh_env_opt_in():
                enabling.append(raw)
        opts_in_by_argument = re.search(r"mesh\s*=\s*True", section) is not None
        assert enabling or opts_in_by_argument, (
            "The Mesh section reads .mesh attributes but names no way to turn mesh on: "
            f"the STRANDS_MESH spellings it shows are {sorted(candidates)}, none of which "
            "opts in, and it never passes mesh=True. Copied as written it raises "
            "AttributeError on None."
        )

    def test_a_bare_robot_leaves_mesh_off(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """The reality the page has to describe."""
        pytest.importorskip("mujoco")
        monkeypatch.delenv("STRANDS_MESH", raising=False)
        mjcf = Path(str(tmp_path)) / "probe.xml"
        mjcf.write_text(_PROBE_MJCF, encoding="utf-8")
        sim = Robot("so100", mode="sim", urdf_path=str(mjcf))
        try:
            assert sim.mesh is None
        finally:
            sim.destroy()
