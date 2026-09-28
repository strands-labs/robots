"""The package's runtime import graph is an acyclic, downward-only DAG.

Two properties of ``strands_robots``, both read from the source by
``scripts/check_import_layers.py`` and both pinned here:

* the **runtime** module-scope import graph has no cycle - a cycle there is what
  makes an import order load-bearing and an interpreter deadlock possible;
* no member is PLACED below its own imports - an inversion has to be one the
  code forces, not one the layer map invented, so a member that could simply
  move up (nothing that imports it sits at or below where it reads) fails
  rather than being declarable;
* every edge that points at a higher layer is written down - a runtime one in
  ``KNOWN_UPWARD_EDGES``, a deferred one (inside a function body) in
  ``KNOWN_DEFERRED_UPWARD_EDGES``. Each pin is an equality, so an inversion
  added to the package fails until someone declares it, and an inversion removed
  from the package fails until someone deletes its line. The rosters are
  ratchets, not suppression lists. Both grade an edge by the layers of its
  two ends, so no module may import the package root, which has no layer:
  ``TestTheContract`` pins that too, or a public name read off the facade
  would be a dependency neither roster can see.

The two properties grade different import kinds because they measure different
things. Acyclicity is about import-time mechanics, so typing-only imports
(``if TYPE_CHECKING:``) and late imports are exempt - they cost nothing on
import and are the two sanctioned ways to break a cycle, which is exactly what
``simulation.base`` and ``simulation.policy_runner`` use them for. Direction is
about who depends on whom, which deferring does not change: a function that
imports the dashboard on its first call still cannot do its job without the
dashboard, so a late import is graded. A typing-only import is not - an
annotation is not a dependency at any point in the run.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "check_import_layers.py"
_PACKAGE_ROOT = _REPO_ROOT / "strands_robots"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_import_layers", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The script is annotated and mypy-clean on its own
# (mypy scripts/check_import_layers.py); it is reached through importlib here
# because scripts/ is not an importable package, so its members are module
# attributes at runtime rather than names mypy can resolve to types.
mod = _load()


@pytest.fixture(scope="module")
def graph() -> Any:
    """The real package graph, built once for every pin that reads the tree."""
    return mod.build_graph(_PACKAGE_ROOT)


class TestTheGraphItIsBuiltFrom:
    """A grader that parses nothing passes every pin below, so measure it first."""

    def test_the_package_is_read_whole(self, graph: Any) -> None:
        assert len(graph.modules) > 250, f"only {len(graph.modules)} modules parsed"
        assert graph.edge_count("runtime") > 500, f"only {graph.edge_count('runtime')} runtime edges"
        assert graph.edge_count("typing_only") > 50
        assert graph.edge_count("late") > 100

    def test_the_layers_are_the_declared_order(self) -> None:
        assert mod.LAYER_NAMES == (
            "core",
            "registry",
            "drivers|mesh",
            "sim|policies",
            "app",
            "tools",
            "dashboard",
        )

    def test_every_top_level_member_has_a_layer(self, graph: Any) -> None:
        assert mod.unassigned_members(graph) == ()

    def test_a_member_missing_from_the_map_is_named(self, graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without this the completeness pin above passes by answering nothing."""
        monkeypatch.setattr(mod, "LAYER_OF_MEMBER", {k: v for k, v in mod.LAYER_OF_MEMBER.items() if k != "drivers"})
        assert mod.unassigned_members(graph) == ("drivers",)


class TestTheParserBehindIt:
    """The three import kinds, and submodule-versus-attribute, on a fake tree.

    Built on disk rather than asserted against ``strands_robots`` so each cell
    isolates one rule: a resolver that answered the parent package for every
    ``from X import y`` would make a leaf's read of a core module look like a
    cycle through ``X/__init__.py``, and nothing in the package's own graph
    distinguishes that from the truth.
    """

    @staticmethod
    def _write(root: Path) -> None:
        (root / "leaf").mkdir(parents=True)
        (root / "__init__.py").write_text("ROOT_ATTR = 1\n", encoding="utf-8")
        (root / "core.py").write_text("VALUE = 2\n", encoding="utf-8")
        (root / "leaf" / "__init__.py").write_text("", encoding="utf-8")
        # Reads a submodule and nothing else, so an edge to the parent package
        # would be visible rather than masked by a second import that earns one.
        (root / "leaf" / "reader.py").write_text(
            "from typing import TYPE_CHECKING\n\n"
            f"from {root.name} import core\n"
            "\n"
            "if TYPE_CHECKING:\n"
            f"    from {root.name}.leaf import writer\n"
            "\n"
            "\n"
            "def late() -> None:\n"
            f"    from {root.name}.leaf import writer as _w\n",
            encoding="utf-8",
        )
        # Reads an attribute of the package root, which is the only way to earn
        # an edge to the package itself.
        (root / "leaf" / "attrs.py").write_text(f"from {root.name} import ROOT_ATTR\n", encoding="utf-8")
        (root / "leaf" / "writer.py").write_text("", encoding="utf-8")

    @pytest.fixture
    def fake(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        root = tmp_path / "fakepkg"
        self._write(root)
        monkeypatch.setattr(mod, "PACKAGE", root.name)
        return mod.build_graph(root)

    @pytest.mark.parametrize(
        ("module", "kind", "expected"),
        [
            ("fakepkg.leaf.reader", "runtime", {"fakepkg.core"}),
            ("fakepkg.leaf.reader", "typing_only", {"fakepkg.leaf.writer"}),
            ("fakepkg.leaf.reader", "late", {"fakepkg.leaf.writer"}),
            ("fakepkg.leaf.attrs", "runtime", {"fakepkg"}),
            ("fakepkg.leaf.attrs", "typing_only", set()),
            ("fakepkg.leaf.attrs", "late", set()),
        ],
    )
    def test_each_import_kind_lands_in_its_own_graph(
        self, fake: Any, module: str, kind: str, expected: set[str]
    ) -> None:
        assert set(getattr(fake, kind).get(module, frozenset())) == expected

    def test_a_two_module_cycle_is_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = tmp_path / "cyclic"
        root.mkdir()
        (root / "__init__.py").write_text("", encoding="utf-8")
        (root / "a.py").write_text("from cyclic import b\n", encoding="utf-8")
        (root / "b.py").write_text("from cyclic import a\n", encoding="utf-8")
        monkeypatch.setattr(mod, "PACKAGE", "cyclic")
        cyclic = mod.build_graph(root)
        assert mod.cycles(cyclic.runtime, frozenset(cyclic.modules)) == [["cyclic.a", "cyclic.b"]]


class TestTheContract:
    """The two properties the roadmap's layered-DAG milestone is measured by."""

    def test_the_runtime_graph_has_no_cycle(self, graph: Any) -> None:
        found = mod.cycles(graph.runtime, frozenset(graph.modules))
        assert found == [], f"runtime import cycles: {found}"

    def test_the_upward_edges_are_exactly_the_declared_ones(self, graph: Any) -> None:
        found = set(mod.upward_edges(graph))
        declared = set(mod.KNOWN_UPWARD_EDGES)
        assert sorted(found - declared) == [], "undeclared inversion; fix it or declare it"
        assert sorted(declared - found) == [], "declared inversion is gone; delete its line"

    def test_the_deferred_upward_edges_are_exactly_the_declared_ones(self, graph: Any) -> None:
        """The same ratchet over the inversions the one above cannot see.

        Deferring an import moves when the dependency is paid, not whether it
        exists, so an inversion inside a function body is an inversion. The ones
        that survive the runtime roster being empty are declared pair by pair,
        which is why "no upward edges" needs this cell to mean what it sounds
        like.
        """
        found = set(mod.upward_edges(graph, "late"))
        declared = set(mod.KNOWN_DEFERRED_UPWARD_EDGES)
        assert sorted(found - declared) == [], "undeclared deferred inversion; fix it or declare it"
        assert sorted(declared - found) == [], "declared deferred inversion is gone; delete its line"

    def test_no_inversion_is_one_a_different_placement_would_remove(self, graph: Any) -> None:
        """A roster entry has to be forced by the code, not by the layer map.

        The two equalities above make a roster a ratchet but say nothing about
        whether an entry deserved to exist. A member placed below where it reads
        reports every downward import as an inversion, and writing those down
        records the placement: ``__main__`` sat in ``app``, so the console entry
        point naming the dashboard CLI it exists to start was an ``app ->
        dashboard`` inversion with a line of its own. Nothing imports a ``python
        -m`` entry point, so it moved to the top layer and the line went.
        """
        assert mod.misplaced_members(graph) == (), (
            "a member is placed below its own imports; move it in LAYERS: "
            f"{[(name, mod.LAYER_NAMES[was], mod.LAYER_NAMES[now]) for name, was, now in mod.misplaced_members(graph)]}"
        )

    def test_the_placement_it_grades_is_the_one_that_was_wrong(
        self, graph: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Put ``__main__`` back in ``app`` and the cell above names it.

        Without this the rule passes by measuring nothing, and it is the only
        placement the package has had that the rule rejects.
        """
        monkeypatch.setitem(mod.LAYER_OF_MEMBER, "__main__", mod.LAYER_NAMES.index("app"))
        assert mod.misplaced_members(graph) == (
            ("__main__", mod.LAYER_NAMES.index("app"), mod.LAYER_NAMES.index("dashboard")),
        )

    @pytest.mark.parametrize(
        ("member", "reads"),
        [
            ("drivers", "policies and simulation"),
        ],
    )
    def test_an_inversion_a_move_cannot_remove_is_not_reported_as_a_placement(
        self, graph: Any, member: str, reads: str
    ) -> None:
        """The one declared inversion is forced, and that is measured.

        ``drivers`` reads a layer above it, so the rule two cells up sees it; it
        is not reported, because something at or below where it reads imports it
        and moving it would only invert that edge instead. Drop the "nothing that
        imports it forbids the move" half of the rule and it turns into a false
        report, which is what this row pins.

        Two members have left this table, and both left the same way - the thing
        they reached up for moved down to them. ``teleop_mixin`` deferred its read
        of the ``teleoperator`` factory, which sat in ``app`` only because it
        borrowed a private lerobot-registry walk from the hardware ``Robot``.
        ``_hitl_audit`` deferred its read of the safety audit log, which sat in
        ``mesh`` only because the mesh wrote to it first - it imports nothing from
        the package, and a ``core`` gate, a ``mesh`` peer and a ``tools`` body all
        write to it. A row here whose member reads nothing above it grades
        nothing, so the row goes when the inversion does.
        """
        assert member in mod.LAYER_OF_MEMBER, reads
        reaching_up = [
            (importer, target)
            for kind in ("runtime", "late")
            for importer, targets in getattr(graph, kind).items()
            if mod.member_of(importer) == member
            for target in targets
            if (mod.layer_of(target) or 0) > mod.LAYER_OF_MEMBER[member]
        ]
        assert reaching_up, f"{member} reads nothing above it, so this row grades nothing"
        assert member not in {name for name, _was, _now in mod.misplaced_members(graph)}

    def test_no_module_reaches_a_public_name_off_the_package_root(self, graph: Any) -> None:
        """The facade is not a back door around the two rosters above.

        ``layer_of`` answers ``None`` for the package root - it re-exports names
        from every layer, so it belongs to none - and :func:`upward_edges` skips
        an edge whose end has no layer. So an import of the root is graded by
        neither equality: planting ``strands_robots.utils`` (``core``) ->
        ``strands_robots`` leaves ``upward_edges`` empty, while 20 of the 46
        lazily re-exported public names resolve into ``tools`` and 4 into
        ``app``. ``from strands_robots import Robot`` in a ``core`` module is
        therefore a dependency on ``app`` that both rosters report as absent.

        Naming the defining module instead gives every internal edge two layers
        and a direction, which is what the rosters ratchet. That the scan can
        see the form at all is
        :meth:`TestTheParserBehindIt.test_each_import_kind_lands_in_its_own_graph`'s
        ``fakepkg.leaf.attrs`` row: reading an attribute off the package is the
        only way to earn an edge to the package itself, so a submodule import
        (``from strands_robots import _dyld``) is not one of these.
        """
        laundered = replace(graph, runtime={**graph.runtime, f"{mod.PACKAGE}.utils": frozenset({mod.PACKAGE})})
        assert mod.upward_edges(laundered) == (), "an edge to the root is graded after all; this pin is redundant"
        offenders = sorted(
            (importer, kind)
            for kind in ("runtime", "typing_only", "late")
            for importer, targets in getattr(graph, kind).items()
            if mod.PACKAGE in targets
        )
        assert offenders == [], "reads a public name off the package root; import the module that defines it"

    def test_the_registry_owns_the_vocabulary_it_validates(self, graph: Any) -> None:
        """A declared field's legal values sit with the loader that refuses the rest.

        ``hardware.driver`` is a registry field, and the loader refuses a value
        outside ``DRIVER_CHOICES`` at load time rather than leaving every reader
        - the factory, a tool, a driver package - to re-check it. The vocabulary
        lived in the driver seam one layer up, so the layer that validates a
        declared entry reached up for the list of what may be declared. The seam
        reads the registry already, so the names went down and both reads now
        point the same way.
        """
        offenders = sorted(
            (importer, target)
            for kind in ("runtime", "typing_only", "late")
            for importer, targets in getattr(graph, kind).items()
            if importer.split(".")[:2] == ["strands_robots", "registry"]
            for target in targets
            if target.split(".")[:2] == ["strands_robots", "drivers"]
        )
        assert offenders == [], f"the registry reaches into the driver seam: {offenders}"
        seam = "strands_robots.drivers.registry"
        assert "strands_robots.registry" in graph.runtime[seam], f"{seam} reads no registry, so this is vacuous"

    def test_the_registry_reads_no_policy_to_import_one(self, graph: Any) -> None:
        """The registry declares providers; the factory imports their classes.

        ``import_policy_class`` walked ``policies.json`` and then fell back to
        scanning ``strands_robots.policies.<name>`` for a
        :class:`~strands_robots.policies.Policy` subclass -- ``issubclass``
        against a class two layers above the registry, so the declarative layer
        deferred an import of the behaviour it is meant to only describe. It
        lives in ``policies.factory`` now, where ``Policy`` is already a
        module-level name, and the registry answers the same question without
        importing anything (``policy_provider_resolves``).

        Graded across all three import kinds: the edge that existed was a late
        import inside the function, which the runtime graph alone does not see.
        """
        offenders = sorted(
            (importer, target)
            for kind in ("runtime", "typing_only", "late")
            for importer, targets in getattr(graph, kind).items()
            if importer.split(".")[:2] == ["strands_robots", "registry"]
            for target in targets
            if target == "strands_robots.policies" or target.startswith("strands_robots.policies.")
        )
        assert offenders == [], f"the registry reaches up for a policy: {offenders}"
        factory = "strands_robots.policies.factory"
        assert "strands_robots.registry" in graph.runtime[factory], f"{factory} reads no registry, so this is vacuous"

    def test_no_driver_imports_a_policy(self, graph: Any) -> None:
        """The cut this contract was first used to make, named on its own.

        A driver that reads a constant out of a policy takes its wire roster from
        that policy's tensor ordering; the roster belongs to the robot.
        """
        drivers_mesh = mod.LAYER_NAMES.index("drivers|mesh")
        sim_policies = mod.LAYER_NAMES.index("sim|policies")
        offenders = [
            edge
            for edge in mod.upward_edges(graph)
            if mod.layer_of(edge[0]) == drivers_mesh and mod.layer_of(edge[1]) == sim_policies
        ]
        assert offenders == []

    def test_no_driver_imports_the_mesh(self, graph: Any) -> None:
        """Drivers and the mesh are siblings: a driver loads no mesh module.

        Loop pacing was the one shared need, and reaching it through
        ``strands_robots.mesh`` ran the mesh package's ``__init__`` - a dozen
        session, security and ROS-bridge modules - for every driver import.
        Graded across all three import kinds, since a deferred import is the
        same dependency on first call.
        """
        offenders = sorted(
            (importer, target)
            for kind in ("runtime", "typing_only", "late")
            for importer, targets in getattr(graph, kind).items()
            if mod.member_of(importer) == "drivers"
            for target in targets
            if mod.member_of(target) == "mesh"
        )
        assert offenders == []

    def test_no_driver_names_a_sibling_package_by_string(self) -> None:
        """A module path handed to ``importlib`` is an import the graph cannot see.

        The Reachy driver resolved its daemon link as
        ``"strands_robots.device_connect.reachy_transport"``, so a driver depended
        on the package 0.7 removes while every edge above read clean. Any string
        constant that is exactly a dotted path into a sibling package counts;
        prose that merely cites one does not, because it is never the whole
        constant.
        """
        sibling = re.compile(r"strands_robots\.(?:mesh|device_connect)(?:\.\w+)*")
        offenders = sorted(
            (path.relative_to(_PACKAGE_ROOT).as_posix(), node.value)
            for path in (_PACKAGE_ROOT / "drivers").rglob("*.py")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and sibling.fullmatch(node.value)
        )
        assert offenders == []

    def test_no_unitree_driver_reaches_into_the_g1_verb_package(self, graph: Any) -> None:
        """The DDS transport three drivers share is theirs, not a verb package's.

        Graded across all three import kinds rather than the runtime graph
        alone: a late import of the transport from inside a driver method is the
        same inversion, deferred to first call, and the Go2's motion-switcher
        read is exactly that shape.
        """
        offenders = sorted(
            (importer, target)
            for kind in ("runtime", "typing_only", "late")
            for importer, targets in getattr(graph, kind).items()
            if importer.startswith("strands_robots.drivers.")
            for target in targets
            if target == "strands_robots.tools.g1" or target.startswith("strands_robots.tools.g1.")
        )
        assert offenders == []

    def test_the_path_sandbox_reads_nothing_from_the_package(self, graph: Any) -> None:
        """A core guard is standard-library-only, which is what lets it sit there.

        ``_path_validation`` is the sandbox every filesystem-writing surface runs
        a caller's directory and file name through - ``training/_validate`` and
        three tool modules - so it belongs under the lowest of them rather than
        beside the one that first needed it. It earns ``core`` by importing
        nothing internal, in any of the three kinds: an internal import here
        would either invert the graph or make this guard's own import order
        load-bearing.
        """
        sandbox = "strands_robots._path_validation"
        assert sandbox in graph.modules
        assert mod.layer_of(sandbox) == mod.LAYER_NAMES.index("core")
        for kind in ("runtime", "typing_only", "late"):
            reads = set(getattr(graph, kind).get(sandbox, frozenset()))
            assert reads == set(), f"{kind} imports from a core guard: {sorted(reads)}"
        importers = {
            importer
            for kind in ("runtime", "typing_only", "late")
            for importer, targets in getattr(graph, kind).items()
            if sandbox in targets
        }
        assert len(importers) >= 4, f"only {sorted(importers)} read the sandbox, so the rule above is vacuous"

    def test_the_operator_gate_sits_below_every_caller_that_asks_a_human(self, graph: Any) -> None:
        """The approval decision is core, not a helper of the package that wrote it.

        ``_command_gate`` is the one owner of the blocklist, the operator
        interrupt and the fail-closed rule; ``_hitl_audit`` is the one owner of
        the row that records the answer. Six tool modules and ``hardware_robot``
        share them, so the pair belongs under the lowest of those callers: a gate
        that lives above one of its callers is a safety decision that caller
        reaches up for, or copies.

        Neither reads anything above ``core``, at import time or from inside a
        call: the empty sets are the pin, so a gate that reaches up for a peer, a
        host or a tool fails here whether it does so on import or on first use.
        The audit row's one dependency used to be the exception - the safety log
        it writes through sat in ``mesh``, so the write was deferred to keep a
        transport stack off the import of a tool that gates a ROS graph. The log
        sits in ``core`` beside it now and imports nothing from the package, so
        the write is a module-scope call and the exception is gone.
        """
        core = mod.LAYER_NAMES.index("core")
        deferred: dict[str, set[str]] = {
            "strands_robots._command_gate": set(),
            "strands_robots._hitl_audit": set(),
        }
        for name, allowed_late in deferred.items():
            assert name in graph.modules
            assert mod.layer_of(name) == core, f"{name} is not in core"
            for kind in ("runtime", "typing_only"):
                above = sorted(t for t in getattr(graph, kind).get(name, frozenset()) if mod.layer_of(t) != core)
                assert above == [], f"{name} has a {kind} import above core: {above}"
            late = {t for t in graph.late.get(name, frozenset()) if mod.layer_of(t) != core}
            assert late == allowed_late, f"{name} defers to {sorted(late)}, not {sorted(allowed_late)}"
        callers = {
            mod.LAYER_NAMES[mod.layer_of(importer)]
            for kind in ("runtime", "typing_only", "late")
            for importer, targets in getattr(graph, kind).items()
            for name in deferred
            if name in targets
        }
        assert {"app", "tools"} <= callers, f"only {sorted(callers)} ask a human, so the rule above is vacuous"

    def test_nothing_below_the_dashboard_reaches_into_it(self, graph: Any) -> None:
        """The web layer is the top of the stack, and deferring a read of it hides that.

        ``dashboard`` is the only layer with nothing above it, so an edge into it
        can only come from below. The equality above grades the runtime graph, and
        every one of these was a late import inside a function, which is why the
        package could report zero inversions while three modules two layers down
        reached up into an OPTIONAL extra for a safety answer: the grant a human's
        yes leaves behind was stored in the dashboard, and ``pose_tool``,
        ``serial_tool`` and the ``Robot`` agent tool each read it through a
        ``try: ... except ImportError: return False``. With the extra installed the
        first gated call imported fastapi, uvicorn, webauthn and PyJWT on the
        motion path; without it, "has a human already said yes?" was answered by a
        failed import. The store sits in ``core`` now
        (:mod:`strands_robots._motion_grants`), under all three of them.

        No edge remains. The last one was ``__main__ -> dashboard.cli``, and the
        command that starts the dashboard has to name its CLI - what was wrong
        was reading that as an inversion, when a ``python -m`` entry point is
        the one module nothing imports and so belongs in the top layer itself.
        """
        dashboard = mod.LAYER_NAMES.index("dashboard")
        offenders = sorted(
            (importer, target)
            for kind in ("runtime", "typing_only", "late")
            for importer, targets in getattr(graph, kind).items()
            for target in targets
            if mod.layer_of(target) == dashboard and (mod.layer_of(importer) or 0) < dashboard
        )
        assert offenders == [], f"a layer below the dashboard imports one of its modules: {offenders}"

    def test_no_layer_below_app_reaches_into_it(self, graph: Any) -> None:
        """The ``app`` layer is a consumer of the package, not a dependency of it.

        ``app`` is where the hosts live - the hardware ``Robot``, the rollout
        runner, teleoperation, recording - so every layer under it exists to be
        composed by one. Three runtime edges pointed the other way, and each was
        a contract stored with its first host rather than under all of them: the
        teleoperation mixin (read by the Device Connect sim driver and the MuJoCo
        ``Simulation`` as well as by ``Robot``) and the recording frame error
        (raised by the dataset writer, caught by the rollout drivers a layer
        down).

        Every kind of import is graded, not just the runtime ones, because a
        deferral is still a dependency: the last edge into ``app`` from below was
        the mixin reaching into the ``teleoperator`` factory from inside
        ``attach_teleop``, and it survived a runtime-only reading of this rule for
        as long as the factory sat in ``app``. It sits in ``drivers|mesh`` now,
        beside the mixin, so the set is empty for all three kinds and a module
        moved back up here fails this cell rather than only the roster equality.
        """
        app = mod.LAYER_NAMES.index("app")
        offenders = sorted(
            (importer, target)
            for kind in ("runtime", "typing_only", "late")
            for importer, targets in getattr(graph, kind).items()
            for target in targets
            if mod.layer_of(target) == app and (below := mod.layer_of(importer)) is not None and below < app
        )
        assert offenders == [], f"a layer below app imports one of its modules: {offenders}"

    @pytest.mark.parametrize(
        ("name", "layer", "deferred_above", "caller_layers"),
        [
            (
                "strands_robots.recording_errors",
                "core",
                frozenset(),
                frozenset({"core", "sim|policies"}),
            ),
            (
                "strands_robots.audit",
                "core",
                frozenset(),
                frozenset({"core", "drivers|mesh", "tools"}),
            ),
            (
                "strands_robots._motion_grants",
                "core",
                frozenset(),
                frozenset({"app", "tools", "dashboard"}),
            ),
            (
                "strands_robots.dataset_metadata",
                "core",
                frozenset(),
                frozenset({"sim|policies", "app", "tools"}),
            ),
            (
                "strands_robots.dataset_source",
                "core",
                frozenset(),
                frozenset({"core", "sim|policies", "tools", "dashboard"}),
            ),
            (
                "strands_robots.streaming_dataset",
                "core",
                frozenset(),
                frozenset({"sim|policies"}),
            ),
            (
                "strands_robots.dataset_transfer",
                "core",
                frozenset(),
                frozenset({"core", "sim|policies"}),
            ),
            (
                "strands_robots.dataset_recorder",
                "core",
                frozenset(),
                frozenset({"sim|policies", "dashboard"}),
            ),
            (
                "strands_robots.teleop_mixin",
                "drivers|mesh",
                frozenset(),
                frozenset({"drivers|mesh", "sim|policies", "app"}),
            ),
            (
                "strands_robots.teleoperator",
                "drivers|mesh",
                frozenset(),
                frozenset({"drivers|mesh", "app", "dashboard"}),
            ),
            (
                "strands_robots.rtps.participant",
                "drivers|mesh",
                frozenset(),
                frozenset({"drivers|mesh", "tools"}),
            ),
            (
                "strands_robots.ros",
                "drivers|mesh",
                frozenset(),
                frozenset({"drivers|mesh", "tools"}),
            ),
            (
                "strands_robots.rosbridge",
                "drivers|mesh",
                frozenset(),
                frozenset({"drivers|mesh", "tools"}),
            ),
            (
                "strands_robots.simulation.recording",
                "sim|policies",
                frozenset(),
                frozenset({"sim|policies", "tools"}),
            ),
        ],
    )
    def test_a_contract_sits_under_every_layer_that_reads_it(
        self,
        graph: Any,
        name: str,
        layer: str,
        deferred_above: frozenset[str],
        caller_layers: frozenset[str],
    ) -> None:
        """Placement, the reads that justify it, and the callers that need it.

        Each row states the same three things the ``_command_gate`` pin above
        states for the operator decision. The module sits in the named layer; it
        reads nothing above that layer at import time, which is what lets it sit
        there; and its callers are the layers that need it, which is why it sits
        under them. A deferred read above the layer is listed explicitly rather
        than allowed in general, so promoting one to module scope has to fail
        here.

        ``teleop_mixin`` and ``teleoperator`` are the pair that row shape was
        needed for. The mixin deferred its read of the factory because the
        factory sat in ``app``, and the only thing holding it there was a private
        walk of lerobot's robot registry that it borrowed from the hardware
        ``Robot`` to tell a caller that ``so101_follower`` is a follower, not a
        leader. That walk is shared from ``core`` now, so the factory sits beside
        the mixin that attaches what it builds, both sets are empty, and the
        mixin's import stays deferred for the reason it states - lerobot - rather
        than for the layering.

        The five ``dataset`` rows are one concern touched five ways: what a
        dataset recorded (``dataset_metadata``, the ``meta/episodes`` parquet the
        sim facade, the ``verify-dataset`` checker and the episode judge each
        certify a run with), which directory a ``repo_id`` names and where an
        episode's frames start (``dataset_source``), the frames streamed back out
        of it (``streaming_dataset``), a finalized directory uploaded to a
        storage bucket (``dataset_transfer``), and the writer that produced it
        (``dataset_recorder``). Four of them lived with the writer in ``app``, so
        a recording backend depended on a CLI and the sim facade's
        ``stream_dataset`` reached up for a module no ``app`` module reads. The
        writer is the fifth: its own imports are ``_dyld``, ``dataset_source``,
        ``dataset_transfer``, ``recording_errors`` and ``utils`` - all ``core`` -
        and nothing in ``app`` reads it, because a recording session exists only
        on the three sim backends a layer below. One caller layer is enough to
        justify a placement - ``streaming_dataset`` has exactly the sim facade,
        the writer exactly the shared recording mixin - and the package root is
        not a layer (``layer_of`` answers ``None`` for it), so its
        ``TYPE_CHECKING`` re-export of a public name is not a caller here.

        ``rtps.participant`` is the DDS mechanics two surfaces share: the
        ``use_rtps`` tool and the ``RtpsRobot`` that drives a ROS 2 base over the
        same wire. They lived in the tool, so the robot imported the ``@tool`` to
        reach a DataWriter. ``ros`` is the same shape one transport over - the
        in-process ``rclpy`` node the ``use_ros`` tool, the ``RosBridgedRobot``
        and the ``AckermannRosRobot`` all publish through - and ``rosbridge`` is
        the same story over a WebSocket, with the ``RosbridgeRobot`` importing
        the ``@tool`` and two of its private names to dial a socket. For each of
        them the ``tools`` caller is what makes the placement load-bearing:
        moving one back up would restore the inversion, and the equality above
        would refuse it.

        ``simulation.recording`` is the row where the empty deferral set is the
        point. The lifecycle every backend mixes in still resolves the recorder
        class inside the call, because that import is what its probe diagnoses: a
        partial install refuses from ``start_recording`` rather than breaking
        ``import strands_robots.simulation``. It carried the last
        ``sim|policies -> app`` inversion until the writer moved under it, and
        the empty set is what stops that edge returning as a deferral.
        """
        assert name in graph.modules
        assert mod.LAYER_NAMES[mod.layer_of(name)] == layer
        index = mod.LAYER_NAMES.index(layer)
        for kind in ("runtime", "typing_only"):
            above = sorted(t for t in getattr(graph, kind).get(name, frozenset()) if mod.layer_of(t) > index)
            assert above == [], f"{name} has a {kind} import above {layer}: {above}"
        late = {t for t in graph.late.get(name, frozenset()) if mod.layer_of(t) > index}
        assert late == set(deferred_above), f"{name} defers to {sorted(late)}, not {sorted(deferred_above)}"
        callers = {
            mod.LAYER_NAMES[index]
            for kind in ("runtime", "typing_only", "late")
            for importer, targets in getattr(graph, kind).items()
            if name in targets
            # The package root re-exports public names under TYPE_CHECKING and
            # has no layer, which is how upward_edges reads it too.
            if (index := mod.layer_of(importer)) is not None
        }
        assert callers == set(caller_layers), f"{name} is read from {sorted(callers)}, not {sorted(caller_layers)}"

    def test_the_script_reports_the_tree_as_conforming(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert mod.main([]) == 0
        assert "OK: no runtime cycle, no undeclared inversion" in capsys.readouterr().out
