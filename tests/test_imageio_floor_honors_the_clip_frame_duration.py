"""The declared imageio floor must ship the clip encoder every extra installs.

:func:`strands_robots.rendering.video.encode_clip` reaches for imageio twice::

    require_optional("imageio", pip_install="imageio imageio-ffmpeg", ...)
    import imageio.v2 as imageio
    ...
    # Pillow's GIF writer takes per-frame duration (ms), not fps.
    imageio.mimsave(str(out), frame_list, duration=1000.0 / int(fps))

That millisecond reading is imageio's from 2.28.0 on; earlier releases read the
same number as seconds. ``[sim-mujoco]`` and ``[sim-isaac]`` declared
``imageio>=2.28.0,<3.0.0`` for it. ``[vera-sim]`` - which ships ``mujoco``, and
whose documented example records its rollout as a GIF - declared a bare
``imageio`` with no bound at all, so the manifest described an install in which
that encoder writes the wrong clip. Measured against the released wheels on
Python 3.12 (the project's minimum), driving the shipped ``encode_clip`` over 12
rendered MuJoCo frames at ``fps=12``, which requests 83.33 ms per frame:

=================  ==========  ===================  ==================
release            imageio.v2  encode_clip          per-frame duration
=================  ==========  ===================  ==================
2.9.0, 2.15.0      absent      ModuleNotFoundError  no clip written
2.16.0             present     AttributeError       no clip written
2.17.0 - 2.19.0    present     RecursionError       no clip written
2.20.0 - 2.27.0    present     returns the clip     83330 ms
2.28.0 - 2.37.4    present     returns the clip     80 ms
=================  ==========  ===================  ==================

Neither failure mode is one the shipped guards can catch. ``require_optional``
takes no minimum version (``module_name``, ``pip_install``, ``extra``,
``purpose``, ``system_install``), so on 2.9.0 it probes the top-level module,
finds it, reports imageio installed - and the very next line raises a bare
``ModuleNotFoundError`` that names no remedy. On 2.20.0 - 2.27.0 nothing raises
at all: the duration is read as seconds, so the clip is encoded 1000x too slow
and a one-second rollout plays for 16 minutes 40 seconds. The file is a valid
12-frame GIF, so ``encode_clip``'s own "the encoder wrote no clip" check passes
and the caller is handed it as a success. The declared range is the only place
either can be refused.

The encoder itself is already covered: the suite pins that an ``encode_clip``
GIF decodes to a per-frame duration matching the requested fps, which holds only
from 2.28.0. What was missing is a floor that admits only the releases where
that can hold. Nothing compared the two because ``[vera-sim]`` is declared in
conflict with ``[all]`` (``[tool.uv] conflicts``) and CI installs ``.[all,dev]``,
so the resolve this extra describes is never built.

:data:`_IMAGEIO_SYMBOL_FLOORS` and :data:`_GIF_DURATION_IN_MILLISECONDS_FROM`
are the single owners of the measurement; the tests below derive the required
floor from them rather than restating a number.
"""

from __future__ import annotations

import ast
import inspect
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import strands_robots
from strands_robots.rendering.video import encode_clip

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent

_DEP = "imageio"
_DISTRIBUTION = "imageio"

#: Sentinel symbol for "the module itself", used by ``import imageio.v2 as ...``
#: and by the ``require_optional("imageio", ...)`` probe.
_MODULE = "<module>"

#: The releases probed to produce the table in the module docstring, oldest first.
_PROBED_RELEASES = (
    "2.9.0",
    "2.15.0",
    "2.16.0",
    "2.17.0",
    "2.18.0",
    "2.19.0",
    "2.20.0",
    "2.21.0",
    "2.22.0",
    "2.25.0",
    "2.26.0",
    "2.27.0",
    "2.28.0",
    "2.29.0",
    "2.31.0",
    "2.34.0",
    "2.37.4",
)

# (module, symbol) -> first release that ships it, measured against the released
# wheels across _PROBED_RELEASES. The ``imageio.v2`` compatibility module lands
# in 2.16.0. The two top-level names predate the oldest release probed, so they
# are recorded at that release rather than at their true origin - only the
# maximum of this table feeds the floor, so an older origin cannot change the
# answer, and an entry can never understate the release the code needs.
_IMAGEIO_SYMBOL_FLOORS: dict[tuple[str, str], str] = {
    ("imageio", _MODULE): "2.9.0",
    ("imageio", "get_writer"): "2.9.0",
    ("imageio.v2", _MODULE): "2.16.0",
    ("imageio.v2", "get_writer"): "2.16.0",
    ("imageio.v2", "mimsave"): "2.16.0",
}

#: First release whose GIF writer reads ``duration`` as the milliseconds
#: ``encode_clip`` passes rather than as seconds. Recorded separately from the
#: symbol table because it is a behaviour, not a name: every release from 2.20.0
#: provides ``imageio.v2.mimsave`` and returns from it normally, having encoded
#: the clip 1000x too slow.
_GIF_DURATION_IN_MILLISECONDS_FROM = "2.28.0"

#: imageio 3.0 has not shipped, so the range above is an audit of the 2.x line
#: only. Every extra states this cap; parity is asserted, the number is not
#: invented here.
_AUDITED_MAJOR_CEILING = "3.0.0"

# A refactor that stops the walk from reaching the sources would make the
# self-maintaining checks vacuous, so pin the size of what it must find.
_MINIMUM_IMPORTED_NAMES = 4


def _required_floor() -> Version:
    """The highest release the imageio names and behaviours the package needs require."""
    return max(Version(v) for v in (*_IMAGEIO_SYMBOL_FLOORS.values(), _GIF_DURATION_IN_MILLISECONDS_FROM))


def _is_dep(module: str) -> bool:
    """True when ``module`` is imageio or a submodule of it."""
    return module == _DEP or module.startswith(_DEP + ".")


def _imported_imageio_names(source: str) -> set[tuple[str, str]]:
    """Every ``(module, symbol)`` one source file reaches for in imageio.

    Covers the three shapes the package uses:

    * ``import imageio.v2 as imageio`` followed by ``imageio.mimsave``, where the
      local name shadows the distribution's own top-level name;
    * ``from imageio import ...``;
    * ``imageio = require_optional("imageio", ...)`` followed by
      ``imageio.get_writer`` - the package's own lazy-optional-import idiom,
      which binds the module through a call rather than an ``import`` statement.

    Only maximal attribute chains are reported, so ``a.b.c`` yields
    ``("<pkg>.b", "c")`` rather than also ``("<pkg>", "b")``.
    """
    tree = ast.parse(source)
    found: set[tuple[str, str]] = set()
    # Local name -> real module path.
    bound: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_dep(alias.name):
                    continue
                found.add((alias.name, _MODULE))
                bound[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level or not _is_dep(node.module or ""):
                continue
            module = node.module or ""
            found.add((module, _MODULE))
            for alias in node.names:
                found.add((module, alias.name))
        elif isinstance(node, ast.Assign):
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "require_optional" or not call.args:
                continue
            requested = call.args[0]
            if not isinstance(requested, ast.Constant) or not isinstance(requested.value, str):
                continue
            if not _is_dep(requested.value):
                continue
            found.add((requested.value, _MODULE))
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound[target.id] = requested.value

    # `a.b.c` is not maximal if it is the receiver of another attribute access.
    receivers = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in receivers:
            continue
        parts: list[str] = []
        cursor: ast.expr = node
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if not isinstance(cursor, ast.Name) or cursor.id not in bound:
            continue
        parts.reverse()
        module = ".".join((bound[cursor.id], *parts[:-1]))
        found.add((module, parts[-1]))
    return found


def _imageio_names_by_file() -> dict[tuple[str, str], list[str]]:
    """Map every imageio name the shipped sources reach for to its files.

    Parses the sources rather than reading ``sys.modules``, so a name reached for
    in a module whose optional deps are absent is still audited.
    """
    found: dict[tuple[str, str], list[str]] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = str(path.relative_to(_PACKAGE_ROOT.parent))
        for name in _imported_imageio_names(path.read_text(encoding="utf-8")):
            found.setdefault(name, []).append(rel)
    return found


def _declared_imageio_specifiers() -> dict[str, Requirement]:
    """Every declared ``imageio`` requirement, keyed by where it lives."""
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    out: dict[str, Requirement] = {}
    for raw in project.get("dependencies", []):
        req = Requirement(raw)
        if req.name == _DISTRIBUTION:
            out["project.dependencies"] = req
    for extra, entries in project.get("optional-dependencies", {}).items():
        for raw in entries:
            req = Requirement(raw)
            if req.name == _DISTRIBUTION:
                out[f"optional-dependencies.{extra}"] = req
    return out


def _declared_floor(req: Requirement) -> Version | None:
    """The lowest release ``req`` admits, or ``None`` when it admits every release."""
    lower = [s for s in req.specifier if s.operator in (">=", "==", "~=")]
    return min(Version(s.version) for s in lower) if lower else None


class TestTheDeclaredFloorCoversTheEncoderEveryExtraShips:
    """Packaging must not admit an imageio too old for the encoder it installs."""

    def test_every_declared_specifier_floors_at_the_capability(self) -> None:
        required = _required_floor()
        specifiers = _declared_imageio_specifiers()
        assert specifiers, "expected at least one declared imageio requirement"

        too_low = {}
        for where, req in specifiers.items():
            floor = _declared_floor(req)
            if floor is None or floor < required:
                too_low[where] = str(req)
        assert not too_low, (
            f"imageio floor must be >= {required}: {_GIF_DURATION_IN_MILLISECONDS_FROM} is the first "
            "release whose GIF writer reads the per-frame duration encode_clip passes as "
            "milliseconds, and the releases below it either raise out of encode_clip or silently "
            f"encode the clip 1000x too slow. These specifiers admit older releases: {too_low}"
        )

    def test_every_declared_specifier_states_the_same_range(self) -> None:
        # An environment resolves one imageio, so a second, looser range
        # describes an install nobody gets - and the code it admits is the same
        # encode_clip in every case.
        ranges = {where: str(req.specifier) for where, req in _declared_imageio_specifiers().items()}
        assert len(set(ranges.values())) == 1, f"every imageio specifier should state the same range; got {ranges}"

    def test_every_declared_specifier_caps_below_the_unaudited_major(self) -> None:
        # _PROBED_RELEASES covers the 2.x line only, so a specifier that admits
        # 3.0 admits a major nothing here has measured.
        uncapped = {
            where: str(req)
            for where, req in _declared_imageio_specifiers().items()
            if req.specifier.contains(_AUDITED_MAJOR_CEILING, prereleases=True)
        }
        assert not uncapped, (
            f"these imageio specifiers admit {_AUDITED_MAJOR_CEILING}, outside the audited 2.x line: {uncapped}"
        )


class TestTheFloorIsSelfMaintaining:
    """The measurement must stay in step with what the sources actually reach for."""

    def test_the_walk_reaches_the_sources(self) -> None:
        # Without this, a walk that silently found nothing would make the two
        # checks below pass while auditing an empty set.
        found = _imageio_names_by_file()
        assert len(found) >= _MINIMUM_IMPORTED_NAMES, (
            f"expected at least {_MINIMUM_IMPORTED_NAMES} imageio names in the shipped sources, "
            f"found {len(found)}: {sorted(f'{m}.{s}' for m, s in found)}"
        )

    def test_every_imageio_import_has_a_recorded_floor(self) -> None:
        imported = _imageio_names_by_file()
        unrecorded = {name: files for name, files in imported.items() if name not in _IMAGEIO_SYMBOL_FLOORS}
        assert not unrecorded, (
            "these imageio names are reached for with no recorded first-shipped release, so nothing "
            f"checks the packaging floor against them: { {f'{m}.{s}': files for (m, s), files in unrecorded.items()} }. "
            "Add each to _IMAGEIO_SYMBOL_FLOORS with the release that first ships it, and raise the "
            "pyproject floor if it is higher."
        )

    def test_every_recorded_release_was_probed(self) -> None:
        # A row invented from a changelog rather than measured against a wheel
        # could name a release that never shipped the name or the behaviour.
        recorded = {*_IMAGEIO_SYMBOL_FLOORS.values(), _GIF_DURATION_IN_MILLISECONDS_FROM}
        unprobed = sorted(v for v in recorded if v not in _PROBED_RELEASES)
        assert not unprobed, (
            f"the measurement records releases that are not in _PROBED_RELEASES: {unprobed}. Probe "
            "the wheel and add it to _PROBED_RELEASES, so every number is measured."
        )

    def test_the_table_records_nothing_the_package_stopped_importing(self) -> None:
        # A stale entry could hold the floor above what the code needs.
        imported = set(_imageio_names_by_file())
        stale = sorted(f"{m}.{s}" for (m, s) in _IMAGEIO_SYMBOL_FLOORS if (m, s) not in imported)
        assert not stale, f"_IMAGEIO_SYMBOL_FLOORS records names the package no longer reaches for: {stale}"

    def test_the_recorded_duration_behaviour_is_still_the_one_encode_clip_relies_on(self) -> None:
        # _GIF_DURATION_IN_MILLISECONDS_FROM is about a specific call: the v2 GIF
        # writer, handed a per-frame duration. Should encode_clip stop making that
        # call, the constant would silently hold the floor for a path nobody takes.
        source = inspect.getsource(encode_clip)
        assert "imageio.v2" in source, (
            "encode_clip no longer imports imageio.v2, so the imageio.v2 rows in "
            "_IMAGEIO_SYMBOL_FLOORS no longer describe what the package needs"
        )
        durations = [
            call
            for call in ast.walk(ast.parse(source.lstrip()))
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "mimsave"
            and any(kw.arg == "duration" for kw in call.keywords)
        ]
        assert durations, (
            "encode_clip no longer passes a per-frame duration to mimsave, so "
            f"_GIF_DURATION_IN_MILLISECONDS_FROM ({_GIF_DURATION_IN_MILLISECONDS_FROM}) no longer describes a release "
            "boundary this package depends on"
        )


class TestTheRecordedNamesExistInTheInstalledImageio:
    """Guard the measurement against imageio removing or moving a name."""

    def test_the_installed_imageio_satisfies_the_declared_range(self) -> None:
        # Otherwise the per-name checks below report on a build no correctly
        # pinned install can get, and a genuine floor violation reads as green.
        installed = Version(pytest.importorskip(_DEP).__version__)
        for where, req in _declared_imageio_specifiers().items():
            assert req.specifier.contains(installed, prereleases=True), (
                f"installed imageio {installed} does not satisfy {where}: {req}"
            )

    def test_the_installed_imageio_reads_the_duration_as_milliseconds(self) -> None:
        # The floor exists to buy this; an installed release below it would make
        # every other check here describe a capability the environment lacks.
        installed = Version(pytest.importorskip(_DEP).__version__)
        assert installed in SpecifierSet(f">={_GIF_DURATION_IN_MILLISECONDS_FROM}"), (
            f"installed imageio {installed} predates {_GIF_DURATION_IN_MILLISECONDS_FROM}, so its GIF "
            "writer reads the duration encode_clip passes as seconds"
        )

    @pytest.mark.parametrize(("module", "symbol"), sorted(_IMAGEIO_SYMBOL_FLOORS))
    def test_name_resolves(self, module: str, symbol: str) -> None:
        imported = pytest.importorskip(module)
        if symbol == _MODULE:
            return
        assert hasattr(imported, symbol), (
            f"{module}.{symbol} is recorded in _IMAGEIO_SYMBOL_FLOORS and reached for by the "
            "package, but the installed imageio does not provide it"
        )
