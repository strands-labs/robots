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


class TestTheNativeDriverRefusalExampleIsStillTrue:
    """The ``driver="strands"`` refusal example must name a robot that has no driver.

    The page shows the refusal verbatim, as a ``>>>`` transcript, so it reads as
    something the reader could paste. That makes it the one block on the page
    whose *premise* can rot without a word of it changing: a robot named here
    because it had no native driver acquires one the day a driver package
    registers it, and the transcript then shows an error that no longer happens
    for a robot that now builds fine.

    It did rot. The example named ``so101``, and ``FeetechDriver`` came to serve
    it - so the block asserted "no native driver is registered for 'so101'"
    while ``Robot("so101", mode="real", driver="strands")`` returned a driver.
    The enumerated list rotted with it, naming two robots when fourteen were
    registered, which is worse than saying nothing: a reader looking up whether
    their robot is natively driven found a list that omitted it and concluded no
    driver existed.

    So the names are graded, not the wording. Every robot the transcript claims
    is natively driven must be, the robot it refuses must have no driver at all,
    and the refusal must still be the one the code raises - checked by raising
    it.

    Correctness is graded, deliberately not completeness. A transcript is a
    capture, and a fifteenth driver leaves it merely older rather than wrong, so
    requiring the list to be exhaustive would fail this page on every driver that
    lands - and the page names ``list_native_drivers()`` as the live answer for
    that reason. A name that is *listed and has no driver* is the failure worth
    catching, because that is the one that sends a reader looking for a driver
    that does not exist.
    """

    @staticmethod
    def _transcript() -> str:
        """Return the fenced block holding the ``driver="strands"`` refusal.

        Returns:
            The block's text.

        Raises:
            AssertionError: If the page ships no such block - the guard would
                otherwise report clean having read nothing.
        """
        text = _DOC.read_text(encoding="utf-8")
        blocks = [
            block
            for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL)
            if 'driver="strands"' in block and "No native driver is registered" in block
        ]
        assert len(blocks) == 1, (
            f"expected exactly one transcript of the driver='strands' refusal, found {len(blocks)}. "
            "This guard grades that block; without it a clean run proves nothing."
        )
        return blocks[0]

    def test_the_refused_robot_really_has_no_native_driver(self) -> None:
        """The premise the transcript rests on, and the one that rotted before."""
        from strands_robots.drivers import get_native_driver_class

        transcript = self._transcript()
        match = re.search(r">>>\s*Robot\(\s*[\"']([^\"']+)[\"']", transcript)
        assert match is not None, f"the transcript shows no Robot(...) call:\n{transcript}"
        name = match.group(1)
        driver_cls = get_native_driver_class(name)
        assert driver_cls is None, (
            f"The page refuses driver='strands' for {name!r} to show what happens when no native "
            f"driver is registered, but {driver_cls.__name__} now serves it: "
            f"Robot({name!r}, mode='real', driver='strands') builds a driver. "
            "Name a robot that still has none, or drop the example."
        )

    def test_the_transcript_is_the_refusal_the_code_raises(self) -> None:
        """Graded by raising it, so a reworded refusal cannot leave the page stale."""
        from strands_robots import Robot

        transcript = self._transcript()
        match = re.search(r">>>\s*Robot\(\s*[\"']([^\"']+)[\"']", transcript)
        assert match is not None
        with pytest.raises(ValueError) as excinfo:
            Robot(match.group(1), mode="real", driver="strands")
        raised = str(excinfo.value)
        for sentence in (
            "No native driver is registered for",
            "so driver='strands' cannot build",
            "Robots with a native driver:",
            "strands_robots.drivers.register_native_driver()",
        ):
            assert sentence in raised, f"the code no longer says {sentence!r}; the page still shows it"
            assert sentence in transcript, f"the page dropped {sentence!r}, which the refusal still says"

    def test_every_robot_the_page_calls_natively_driven_really_is(self) -> None:
        """A name in the list must have a driver, or it sends a reader to a dead end."""
        from strands_robots.drivers import get_native_driver_class, list_native_drivers

        transcript = self._transcript()
        listed = re.search(r"Robots with a native driver:\s*(.*?)\.\s", transcript, re.DOTALL)
        assert listed is not None, f"the transcript shows no list of natively driven robots:\n{transcript}"
        names = [
            candidate
            for candidate in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", listed.group(1))
            # The elision marker and any prose in the tail are not robot names.
            if candidate in set(list_native_drivers()) or get_native_driver_class(candidate) is None
        ]
        assert names, f"no robot name was read out of {listed.group(1)!r}; the guard would prove nothing"
        wrong = [name for name in names if get_native_driver_class(name) is None]
        assert not wrong, (
            f"The page lists {wrong} among robots with a native driver, but none is registered for them. "
            "A reader checking whether their robot is natively driven is told yes and finds nothing."
        )

    def test_the_page_names_the_live_listing_helper(self) -> None:
        """A captured list is only honest if the page says where the live one is."""
        assert "list_native_drivers()" in _DOC.read_text(encoding="utf-8"), (
            "The transcript's list of natively driven robots is a capture, so the page must name "
            "list_native_drivers() as the way to get the current one."
        )
