"""The ``max_steps`` horizon domain on every LiberoAdapter construction surface.

``LiberoAdapter.__init__`` bounds ``max_steps`` with the shared strict-count
domain, and its own comment states what the bound is for: the value becomes the
benchmark's per-episode ``range(max_steps)`` bound, so a zero or negative
horizon runs episodes of zero length that still report a 0% success rate, and an
``int()`` coercion alone would truncate ``2.7`` to 2 and read ``True`` as 1.

Four public surfaces carry the parameter -- the constructor plus
:meth:`~strands_robots.benchmarks.libero.LiberoAdapter.from_text`,
:meth:`~strands_robots.benchmarks.libero.LiberoAdapter.from_file` and
:func:`~strands_robots.benchmarks.libero.load_libero_suite`, the last three
forwarding into the constructor -- and the refusal reaches the caller through
two different channels:

* the three adapter surfaces raise ``ValueError`` carrying the shared domain's
  verdict verbatim;
* the suite loader wraps each ``from_file`` in a ``except ... ValueError``
  whose contract is to skip a malformed task and continue, so an unusable
  horizon registers **no** task, returns normally, and reports the refusal once
  per task file at WARNING level.

The sibling ``init_jitter`` refusal one branch above is exercised by the
existing suite; this one was not, on any surface, so the domain's whole
rejecting half was unverified. These tests drive it on all four and pin the
wording against the shared helper rather than against a local copy, so a
re-worded or widened domain cannot pass unnoticed.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from strands_robots.benchmarks.libero import LiberoAdapter, load_libero_suite, parse_bddl
from strands_robots.utils import positive_count_error, positive_whole_number_error

_BDDL = "(define (problem pick_cube) (:goal (grasped cube_1)))"

# Keep construction free of scene generation and camera installs: this module is
# about the horizon parameter, and neither step is reached before the guard.
_INERT: dict[str, Any] = {"auto_generate_scene": False, "install_cameras": False}

#: Values no ``range()`` bound can consume. ``None`` is deliberately absent --
#: it is the documented "not supplied" spelling and is pinned separately.
UNUSABLE = [
    pytest.param(0, id="zero"),
    pytest.param(-5, id="negative"),
    pytest.param(True, id="True"),
    pytest.param(False, id="False"),
    pytest.param(2.7, id="fractional"),
    pytest.param(3.0, id="integral-float"),
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="inf"),
    pytest.param("10", id="numeric-string"),
    pytest.param([4], id="list"),
]

#: Values the *wider* whole-number domain admits and the strict count domain
#: does not. They separate the two shared helpers, so a widened domain fails.
STRICT_ONLY = [
    pytest.param(3.0, id="integral-float"),
    pytest.param(np.int64(7), id="numpy-int"),
    pytest.param(np.float64(4.0), id="numpy-integral-float"),
]


def _via_init(value: Any, tmp_path: pathlib.Path) -> LiberoAdapter:
    return LiberoAdapter(parse_bddl(_BDDL), max_steps=value, **_INERT)


def _via_from_text(value: Any, tmp_path: pathlib.Path) -> LiberoAdapter:
    return LiberoAdapter.from_text(_BDDL, max_steps=value, **_INERT)


def _via_from_file(value: Any, tmp_path: pathlib.Path) -> LiberoAdapter:
    path = tmp_path / "task.bddl"
    path.write_text(_BDDL)
    return LiberoAdapter.from_file(path, max_steps=value, **_INERT)


#: The three surfaces whose refusal channel is a raise.
RAISING_SURFACES: list[Any] = [
    pytest.param(_via_init, id="__init__"),
    pytest.param(_via_from_text, id="from_text"),
    pytest.param(_via_from_file, id="from_file"),
]


def _suite_dir(tmp_path: pathlib.Path, tasks: tuple[str, ...] = ("task_a", "task_b")) -> pathlib.Path:
    """A ``libero_spatial`` directory holding one well-formed BDDL per task."""
    suite_dir = tmp_path / "libero_spatial"
    suite_dir.mkdir()
    for task in tasks:
        (suite_dir / f"{task}.bddl").write_text(_BDDL)
    return suite_dir


def _load(suite_dir: pathlib.Path, value: Any) -> dict[str, LiberoAdapter]:
    return load_libero_suite(
        "libero_spatial",
        bddl_dir=suite_dir,
        max_steps=value,
        load_init_states=False,
        adapter_kwargs=dict(_INERT),
    )


class TestEveryAdapterSurfaceRefusesAnUnusableHorizon:
    """The constructor and both classmethods refuse, with the shared wording."""

    @pytest.mark.parametrize("build", RAISING_SURFACES)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_refusal_carries_the_shared_domains_verdict(
        self, build: Callable[[Any, pathlib.Path], LiberoAdapter], value: Any, tmp_path: pathlib.Path
    ) -> None:
        """Message equality, not a substring: a locally re-worded copy would drift."""
        expected = positive_count_error(value, "max_steps", "LiberoAdapter")
        assert expected is not None, "probe value must be outside the domain"
        with pytest.raises(ValueError) as excinfo:
            build(value, tmp_path)
        assert str(excinfo.value) == expected


class TestTheSuiteLoaderNamesTheValueAndRegistersNothing:
    """The loader's channel differs: it skips, continues and reports per task."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_no_task_registers_and_every_report_names_the_value(
        self, value: Any, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        suite_dir = _suite_dir(tmp_path)
        with caplog.at_level("WARNING"):
            registered = _load(suite_dir, value)
        assert registered == {}
        skipped = [rec.getMessage() for rec in caplog.records if "Skipping LIBERO task" in rec.getMessage()]
        assert len(skipped) == 2, skipped
        expected = positive_count_error(value, "max_steps", "LiberoAdapter")
        assert expected is not None, "probe value must be outside the domain"
        assert all(expected in message for message in skipped), skipped


class TestAUsableHorizonIsStillHonored:
    """The over-reach control: nothing the domain admits became harder to pass."""

    @pytest.mark.parametrize("build", RAISING_SURFACES)
    def test_a_positive_integer_is_stored_verbatim(
        self, build: Callable[[Any, pathlib.Path], LiberoAdapter], tmp_path: pathlib.Path
    ) -> None:
        assert build(250, tmp_path).max_steps == 250

    def test_the_loader_registers_every_task_with_the_horizon(self, tmp_path: pathlib.Path) -> None:
        registered = _load(_suite_dir(tmp_path), 250)
        assert sorted(registered) == ["libero-spatial-task_a", "libero-spatial-task_b"]
        assert {adapter.max_steps for adapter in registered.values()} == {250}

    @pytest.mark.parametrize("build", RAISING_SURFACES)
    def test_none_means_not_supplied_and_keeps_the_class_default(
        self, build: Callable[[Any, pathlib.Path], LiberoAdapter], tmp_path: pathlib.Path
    ) -> None:
        """The guard is gated on ``is not None``, so ``None`` is not a refusal."""
        assert build(None, tmp_path).max_steps == LiberoAdapter.max_steps


class TestTheStrictCountDomainIsTheOneApplied:
    """A widened domain would admit these, so the choice of helper is pinned."""

    @pytest.mark.parametrize("value", STRICT_ONLY)
    def test_a_value_only_the_whole_number_domain_admits_is_refused(self, value: Any, tmp_path: pathlib.Path) -> None:
        assert positive_whole_number_error(value, "max_steps", "LiberoAdapter") is None
        with pytest.raises(ValueError, match="max_steps"):
            _via_from_text(value, tmp_path)


def _max_steps_surfaces() -> dict[str, tuple[bool, bool]]:
    """Map each public ``max_steps`` surface to ``(guards, forwards)``.

    Rooted at the package that defines :class:`LiberoAdapter` rather than a path
    literal, so the scan follows a rename.
    """
    root = pathlib.Path(inspect.getfile(LiberoAdapter)).parent
    found: dict[str, tuple[bool, bool]] = {}
    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef):
                continue
            # A constructor is a public surface; every other private helper
            # receives an already-validated value.
            if node.name.startswith("_") and node.name != "__init__":
                continue
            if "max_steps" not in [a.arg for a in node.args.args + node.args.kwonlyargs]:
                continue
            segment = ast.get_source_segment(source, node) or ""
            guards = "positive_count_error(max_steps" in segment
            forwards = "max_steps=max_steps" in segment
            found[f"{path.name}::{node.name}"] = (guards, forwards)
    return found


class TestNoMaxStepsSurfaceDrifts:
    """A fifth surface cannot start reading the horizon without the domain."""

    def test_the_surfaces_are_the_four_known_ones(self) -> None:
        assert set(_max_steps_surfaces()) == {
            "adapter.py::__init__",
            "adapter.py::from_file",
            "adapter.py::from_text",
            "suite.py::load_libero_suite",
        }

    def test_every_surface_either_guards_or_forwards(self) -> None:
        adrift = {name for name, (guards, forwards) in _max_steps_surfaces().items() if not (guards or forwards)}
        assert not adrift, f"{sorted(adrift)} read max_steps without the shared domain"

    def test_the_owner_is_the_constructor(self) -> None:
        owners = {name for name, (guards, _forwards) in _max_steps_surfaces().items() if guards}
        assert owners == {"adapter.py::__init__"}
