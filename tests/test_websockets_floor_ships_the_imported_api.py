"""The declared websockets floor must ship the websockets API the package imports.

``strands_robots/inference/server.py`` annotates ``PolicyServer._server`` with
``websockets.sync.server.Server``::

    if TYPE_CHECKING:
        from websockets.sync.server import Server, ServerConnection
    ...
        self._server: Server | None = None

websockets 12.0 has no such name - it spells that class ``WebSocketServer``, and
``Server`` first appears in 13.0. Both ``[inference]`` and ``[cosmos3-service]``
declared ``websockets>=12.0``, so the manifest described an install in which the
package's own annotation does not resolve. Measured against the released wheels
(``Server`` is the only name the declared floor was missing):

=============  ================================================
release        names missing of the nine the sources reach for
=============  ================================================
9.1, 10.x      all seven ``websockets.sync.*`` names
11.0 - 12.0    ``websockets.sync.server.Server``
13.0 - 17.0.1  none
=============  ================================================

A floor is not only about names. From 13.0 through 16.x,
``websockets.sync.server.Server.shutdown()`` closed the listening socket and
nothing else, so ``PolicyServer.stop()`` returned while a client that was already
connected went on streaming observations in and receiving action chunks back -
the wrapped policy still being invoked, and on a robot still driving the arm,
after the caller was told the server stopped. 17.0 closes the connections it
accepted (code 1001) and returns only once every connection handler has
terminated, which is what makes that teardown hold. Measured against the released
wheels, on the unchanged sources:

=============  ================================================================
release        a client connected when ``stop()`` is called
=============  ================================================================
16.1.1         still answered with action chunks afterwards
17.0, 17.1     refused - the connection is closed and its handler joined
=============  ================================================================

So the declared floor is the maximum of two tables: the release that first ships
each *name* the sources import (:data:`_WEBSOCKETS_SYMBOL_FLOORS`) and the
release that first ships each *behaviour* they rely on
(:data:`_WEBSOCKETS_BEHAVIOUR_FLOORS`). The second is what carries the floor
today, and the property itself is graded from a client's point of view by
tests/inference/test_a_stopped_server_stops_serving_its_clients.py.

Nothing caught the 12.0 hole because the import lives under ``TYPE_CHECKING`` and
is never executed: on 12.0 the package imports, binds a port, serves and stops cleanly.
Only a type checker notices, and it reports the dependency by name -
``error: Module "websockets.sync.server" has no attribute "Server"
[attr-defined]`` - so a 12.0 environment fails the type gate for a reason no
source change can fix. CI type-checks the locked resolve (16.0 today), never the
declared floor, and nothing compared what the manifest admits against what the
sources import.

:data:`_WEBSOCKETS_SYMBOL_FLOORS` is the single owner of the measurement. The
tests below derive the required floor from it rather than restating a number, and
:meth:`TestTheFloorIsSelfMaintaining.test_every_websockets_import_has_a_recorded_floor`
fails the moment the package reaches for a websockets name that is not in the
table - so a future use of a newer websockets API cannot silently leave the floor
behind.

No upper bound is asserted. Capping ``websockets`` narrows every consumer's
resolve and is a separate call from correcting a floor that is measurably wrong;
the table above records the releases that were probed, not a supported range.
"""

from __future__ import annotations

import ast
import inspect
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import strands_robots

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent

_DEP = "websockets"
_DISTRIBUTION = "websockets"

#: Sentinel symbol for "the module itself", used by a bare ``import websockets``
#: or by ``from websockets.sync.server import ...`` (which needs the module too).
_MODULE = "<module>"

#: The releases probed to produce the table below, oldest first.
_PROBED_RELEASES = (
    "9.1",
    "10.0",
    "10.4",
    "11.0",
    "11.0.3",
    "12.0",
    "13.0",
    "13.1",
    "14.0",
    "15.0",
    "16.0",
    "16.1.1",
    "17.0",
    "17.0.1",
    "17.1",
)

# (module, symbol) -> first release that ships it, measured against the released
# wheels across _PROBED_RELEASES. The synchronous implementation
# (``websockets.sync``) lands in 11.0 and ``Server`` joins it in 13.0. The two
# ``websockets``-level names predate the oldest release probed, so they are
# recorded at that release rather than at their true origin - only the maximum of
# this table feeds the floor, so an older origin cannot change the answer, and an
# entry can never understate the release the code needs.
_WEBSOCKETS_SYMBOL_FLOORS: dict[tuple[str, str], str] = {
    ("websockets", _MODULE): "9.1",
    ("websockets", "connect"): "9.1",
    ("websockets.sync.client", _MODULE): "11.0",
    ("websockets.sync.client", "ClientConnection"): "11.0",
    ("websockets.sync.client", "connect"): "11.0",
    ("websockets.sync.server", _MODULE): "11.0",
    ("websockets.sync.server", "Server"): "13.0",
    ("websockets.sync.server", "ServerConnection"): "11.0",
    ("websockets.sync.server", "serve"): "11.0",
}

#: Behaviour -> first release that ships it, measured against the released wheels
#: across _PROBED_RELEASES by running the teardown tests on the unchanged
#: sources. A name resolving is not enough for these: the name existed all along
#: and only what it does changed, so nothing but the floor can express the
#: requirement. Each key names the observable the release is identified by, and
#: :class:`TestTheInstalledWebsocketsShipsTheRequiredBehaviour` checks that the
#: installed build actually has it - so a floor recorded here cannot be a claim
#: about a wheel nobody probed.
_WEBSOCKETS_BEHAVIOUR_FLOORS: dict[str, str] = {
    # PolicyServer.stop() and a returning serve() both call Server.shutdown()
    # and are documented to stop the server serving rather than listening.
    # Through 16.x shutdown() closed the listening socket alone; 17.0 closes the
    # accepted connections too and waits for their handlers, and it is the
    # `close_connections` parameter that arrives with that change.
    "websockets.sync.server.Server.shutdown closes accepted connections (close_connections)": "17.0",
}

# A refactor that stops the walk from reaching the sources would make the
# self-maintaining checks vacuous, so pin the size of what it must find.
_MINIMUM_IMPORTED_NAMES = 5


def _required_floor() -> Version:
    """The highest first-shipped release among the names and behaviours used."""
    return max(Version(v) for v in (*_WEBSOCKETS_SYMBOL_FLOORS.values(), *_WEBSOCKETS_BEHAVIOUR_FLOORS.values()))


def _requirements_at(release: Version) -> list[str]:
    """Every name and behaviour whose first-shipped release is ``release``."""
    names = [f"{m}.{s}" for (m, s), v in _WEBSOCKETS_SYMBOL_FLOORS.items() if Version(v) == release]
    behaviours = [k for k, v in _WEBSOCKETS_BEHAVIOUR_FLOORS.items() if Version(v) == release]
    return sorted(names + behaviours)


def _is_dep(module: str) -> bool:
    """True when ``module`` is websockets or a submodule of it."""
    return module == _DEP or module.startswith(_DEP + ".")


def _imported_websockets_names(source: str) -> set[tuple[str, str]]:
    """Every ``(module, symbol)`` one source file reaches for in websockets.

    Covers the three shapes the package uses:

    * ``from websockets.sync.server import Server`` - including under
      ``TYPE_CHECKING``, which is where the name that set this floor lives;
    * ``import websockets`` followed by a dotted use (``websockets.connect``);
    * ``import websockets.sync.client as _wsc`` followed by ``_wsc.connect``.

    Only maximal attribute chains are reported, so ``a.b.c`` yields
    ``("<pkg>.b", "c")`` rather than also ``("<pkg>", "b")``.
    """
    tree = ast.parse(source)
    found: set[tuple[str, str]] = set()
    # Local name -> real module path, for both `import websockets` and
    # `import websockets.sync.client as _wsc`.
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


def _websockets_names_by_file() -> dict[tuple[str, str], list[str]]:
    """Map every websockets name the shipped sources reach for to its files.

    Parses the sources rather than reading ``sys.modules``, so a name used only
    under ``TYPE_CHECKING`` - or in a module whose optional deps are absent - is
    still audited.
    """
    found: dict[tuple[str, str], list[str]] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = str(path.relative_to(_PACKAGE_ROOT.parent))
        for name in _imported_websockets_names(path.read_text(encoding="utf-8")):
            found.setdefault(name, []).append(rel)
    return found


def _declared_websockets_specifiers() -> dict[str, Requirement]:
    """Every declared ``websockets`` requirement, keyed by where it lives."""
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


class TestTheDeclaredFloorCoversEveryImportedName:
    """Packaging must not admit a websockets too old for the code it installs."""

    def test_every_declared_specifier_floors_at_the_capability(self) -> None:
        required = _required_floor()
        specifiers = _declared_websockets_specifiers()
        assert specifiers, "expected at least one declared websockets requirement"

        too_low = {}
        for where, req in specifiers.items():
            lower = [s for s in req.specifier if s.operator in (">=", "==", "~=")]
            assert lower, f"{where}: {req} declares no lower bound"
            floor = min(Version(s.version) for s in lower)
            if floor < required:
                too_low[where] = str(req)
        assert not too_low, (
            f"websockets floor must be >= {required} because the package requires "
            f"{_requirements_at(required)}; these specifiers admit older releases: {too_low}"
        )

    def test_the_floors_agree_across_every_extra(self) -> None:
        # An environment resolves one websockets, so two different floors would
        # leave the lower one describing an install nobody gets.
        floors = {
            where: min(Version(s.version) for s in req.specifier if s.operator in (">=", "==", "~="))
            for where, req in _declared_websockets_specifiers().items()
        }
        assert len(set(floors.values())) == 1, f"every websockets specifier should declare the same floor; got {
            {w: str(v) for w, v in floors.items()}
        }"


class TestTheFloorIsSelfMaintaining:
    """The table must stay in step with what the sources actually reach for."""

    def test_the_walk_reaches_the_sources(self) -> None:
        # Without this, a walk that silently found nothing would make the two
        # checks below pass while auditing an empty set.
        found = _websockets_names_by_file()
        assert len(found) >= _MINIMUM_IMPORTED_NAMES, (
            f"expected at least {_MINIMUM_IMPORTED_NAMES} websockets names in the shipped "
            f"sources, found {len(found)}: {sorted(f'{m}.{s}' for m, s in found)}"
        )

    def test_every_websockets_import_has_a_recorded_floor(self) -> None:
        imported = _websockets_names_by_file()
        unrecorded = {name: files for name, files in imported.items() if name not in _WEBSOCKETS_SYMBOL_FLOORS}
        assert not unrecorded, (
            "these websockets names are reached for with no recorded first-shipped release, "
            "so nothing checks the packaging floor against them: "
            f"{ {f'{m}.{s}': files for (m, s), files in unrecorded.items()} }. "
            "Add each to _WEBSOCKETS_SYMBOL_FLOORS with the release that first ships it, "
            "and raise the pyproject floor if it is higher."
        )

    def test_every_recorded_release_was_probed(self) -> None:
        # A row invented from a changelog rather than measured against a wheel
        # could name a release that never shipped the name.
        recorded = (*_WEBSOCKETS_SYMBOL_FLOORS.values(), *_WEBSOCKETS_BEHAVIOUR_FLOORS.values())
        unprobed = sorted({v for v in recorded if v not in _PROBED_RELEASES})
        assert not unprobed, (
            f"the floor tables record releases that are not in _PROBED_RELEASES: {unprobed}. "
            "Probe the wheel and add it to _PROBED_RELEASES, so every number in the table is measured."
        )

    def test_the_table_records_nothing_the_package_stopped_importing(self) -> None:
        # A stale entry could hold the floor above what the code needs.
        imported = set(_websockets_names_by_file())
        stale = sorted(f"{m}.{s}" for (m, s) in _WEBSOCKETS_SYMBOL_FLOORS if (m, s) not in imported)
        assert not stale, f"_WEBSOCKETS_SYMBOL_FLOORS records names the package no longer reaches for: {stale}"


class TestTheInstalledWebsocketsShipsTheRequiredBehaviour:
    """A behaviour floor is only worth the observable that identifies it."""

    def test_shutdown_takes_the_close_connections_parameter(self) -> None:
        # The parameter arrives with the change that closes accepted connections
        # on shutdown, so its absence is how a downgrade below the recorded floor
        # is caught here rather than in a robot's teardown. The default is the
        # half that matters: PolicyServer calls shutdown() with no arguments.
        server_module = pytest.importorskip("websockets.sync.server")
        parameter = inspect.signature(server_module.Server.shutdown).parameters.get("close_connections")
        assert parameter is not None, (
            "the installed websockets Server.shutdown() takes no close_connections parameter, so it is "
            "older than the floor recorded in _WEBSOCKETS_BEHAVIOUR_FLOORS and closes the listening "
            "socket alone - PolicyServer.stop() would return while a connected client is still served"
        )
        assert parameter.default is True, (
            f"PolicyServer calls Server.shutdown() with no arguments, so closing the accepted "
            f"connections has to be the default; got {parameter.default!r}"
        )


class TestTheRecordedNamesExistInTheInstalledWebsockets:
    """Guard the table against websockets removing or moving a name."""

    def test_the_installed_websockets_satisfies_the_declared_floor(self) -> None:
        # Otherwise the per-name checks below report on a build no correctly
        # pinned install can get, and a genuine floor violation reads as green.
        installed = Version(pytest.importorskip(_DEP).__version__)
        for where, req in _declared_websockets_specifiers().items():
            assert req.specifier.contains(installed, prereleases=True), (
                f"installed websockets {installed} does not satisfy {where}: {req}"
            )

    @pytest.mark.parametrize(("module", "symbol"), sorted(_WEBSOCKETS_SYMBOL_FLOORS))
    def test_name_resolves(self, module: str, symbol: str) -> None:
        imported = pytest.importorskip(module)
        if symbol == _MODULE:
            return
        assert hasattr(imported, symbol), (
            f"{module}.{symbol} is recorded in _WEBSOCKETS_SYMBOL_FLOORS and reached for by "
            "the package, but the installed websockets does not provide it"
        )
