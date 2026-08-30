"""Tests for :mod:`strands_robots.tools.g1.g1_slam_relocalize_envelope`.

The module ports the neon SLAM runner's ``_try_relocalize`` gate
(``cagataycali/neon-the-g1/tools/g1_slam.py``) into a read-only
lookup pair.  The tests grade three things: import hygiene (no
optional SLAM stack loads at import), snapshot fidelity (the
envelope carries every neon-observed constant on both verbs), and
the admit/refuse decision matrix for the three graded dimensions
(fitness, translation magnitude, rotation trace).

The refusals share one module-local :data:`_REFUSAL_TEXT`, so a
misread of any one dimension is graded by both the dimension-specific
cell and by a cross-cell that a multi-dimension violation reports
every refusal at once -- the neon runner short-circuits on the first
refusal, but a caller planning a relocalise wants every violated
bound named so the next attempt fixes the whole triple.

Refs strands-labs/robots#358.
"""

from __future__ import annotations

import importlib
import math
import sys

import pytest

MODULE_PATH = "strands_robots.tools.g1.g1_slam_relocalize_envelope"


class TestTheImportPullsNoOptionalSlamModule:
    """The module docstring's import-hygiene contract, refs strands-labs/robots#358.

    A caller authoring a relocalise plan before any SLAM extra is
    installed on their host still gets the envelope back verbatim;
    the module's advertised no-optional-import property is asserted
    against the process's own :data:`sys.modules` after import.
    """

    def test_the_import_pulls_no_unitree_sdk2py_submodule(self) -> None:
        # Snapshot before importing so a submodule loaded by an
        # earlier test does not tar this one; only the delta this
        # module's import introduces is graded.  Pop the target
        # first so the delta is the module's own import cost even
        # when a prior test has already loaded it.
        sys.modules.pop(MODULE_PATH, None)
        before = set(sys.modules)
        importlib.import_module(MODULE_PATH)
        added = set(sys.modules) - before
        leaked = {name for name in added if "unitree" in name.lower()}
        assert leaked == set(), (
            f"the import of {MODULE_PATH} pulled unitree_sdk2py "
            f"submodules {sorted(leaked)}; the neon bundle's SLAM "
            "gate ports as a lookup, not as an SDK call"
        )

    def test_the_import_pulls_no_new_optional_slam_submodule(self) -> None:
        # numpy / open3d / kiss_icp are the SLAM extra the neon
        # runner's own _try_relocalize reaches.  numpy (top-level)
        # may already be in sys.modules from an earlier stdlib or
        # test-fixture import, so the cell checks the *delta*: a
        # fresh submodule newly imported by this module's load path
        # is a rule violation; an already-loaded numpy from an
        # unrelated pathway is not.  Pinned so a future edit that
        # reaches into e.g. numpy.linalg for a compile-time
        # translation-norm ceiling fails this cell first.
        sys.modules.pop(MODULE_PATH, None)
        before = set(sys.modules)
        importlib.import_module(MODULE_PATH)
        added = set(sys.modules) - before
        # Match a submodule name (contains a dot after the top
        # name) OR a fresh top-level name; already-loaded top-level
        # numpy/open3d/kiss_icp are absent from ``added`` by
        # definition of the delta so they cannot fail this cell.
        leaked = {name for name in added if name.split(".")[0] in ("numpy", "open3d", "kiss_icp")}
        assert leaked == set(), (
            f"the import of {MODULE_PATH} newly pulled optional SLAM "
            f"submodules {sorted(leaked)}; the envelope is a "
            "module-level constant snapshot and reaches none of them. "
            "The open3d / kiss_icp / numpy calls belong inside a "
            "future driver-side wrapper, refs strands-labs/robots#358."
        )


class TestTheEnvelopeSnapshotIsFaithful:
    """The envelope descriptor mirrors the neon runner's own bounds byte-for-byte."""

    def test_the_envelope_lists_every_neon_runner_constant(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_list_slam_relocalize_envelope,
        )

        payload = g1_list_slam_relocalize_envelope()
        assert payload["status"] == "success"
        envelope = payload["envelope"]
        # Every field the module docstring names is present on the
        # envelope; a widen to the descriptor lands in one place.
        assert set(envelope) == {
            "fitness_min",
            "fitness_max",
            "translation_max_m",
            "rotation_trace_min",
            "rotation_trace_max",
        }
        # The values are the neon-runner-observed constants.
        assert envelope["fitness_min"] == 0.3
        assert envelope["fitness_max"] == 1.0
        assert envelope["translation_max_m"] == 50.0
        assert envelope["rotation_trace_min"] == 0.0
        assert envelope["rotation_trace_max"] == 3.0

    def test_the_envelope_carries_the_refusal_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_list_slam_relocalize_envelope,
        )

        payload = g1_list_slam_relocalize_envelope()
        refusals = payload["refusals"]
        assert len(refusals) == 1
        # The single refusal descriptor carries a text field naming
        # the surface (relocalise gate) and citing the umbrella
        # issue so a caller reading the refusal can find the
        # driver-side follow-up.
        text = refusals[0]["text"]
        assert "relocalise" in text
        assert "strands-labs/robots#358" in text

    def test_the_admits_verb_reports_the_same_envelope(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_list_slam_relocalize_envelope,
            g1_slam_relocalize_admits,
        )

        list_env = g1_list_slam_relocalize_envelope()["envelope"]
        admits_env = g1_slam_relocalize_admits()["envelope"]
        # A widen to the descriptor lands in one place because both
        # verbs read the same _envelope() helper.
        assert list_env == admits_env


class TestTheDefaultTripleAdmits:
    """The verb's default arguments land on the admitted side of every clamp."""

    def test_the_default_triple_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits()
        assert payload["status"] == "success"
        assert payload["admits"] is True
        assert payload["refusals"] == []


class TestTheFitnessDimensionIsGraded:
    """Fitness clamp: floor at 0.3 inclusive, ceiling at 1.0 inclusive."""

    def test_a_fitness_at_the_floor_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(fitness=0.3)
        assert payload["admits"] is True

    def test_a_fitness_below_the_floor_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(fitness=0.29)
        assert payload["admits"] is False
        assert len(payload["refusals"]) == 1
        r = payload["refusals"][0]
        assert r["dimension"] == "fitness"
        assert r["bound_key"] == "fitness_min"
        assert r["bound"] == 0.3
        assert r["comparison"] == "value < bound"
        assert "strands-labs/robots#358" in r["text"]

    def test_a_fitness_at_the_ceiling_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(fitness=1.0)
        assert payload["admits"] is True

    def test_a_fitness_above_the_ceiling_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(fitness=1.5)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "fitness"
        assert r["bound_key"] == "fitness_max"
        assert r["comparison"] == "value > bound"

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_fitness_refuses(self, value: float) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(fitness=value)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "fitness"
        assert r["comparison"] == "non-finite"

    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_fitness_refuses(self, value: bool) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        # True is int(1) which would otherwise silently look up 1.0
        # (a legitimate perfect-fit fitness). Naming the refusal at
        # the boundary surfaces the type mistake.
        payload = g1_slam_relocalize_admits(fitness=value)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "fitness"
        assert r["comparison"] == "non-int"


class TestTheTranslationDimensionIsGraded:
    """Translation-magnitude clamp: 0.0 floor (shape), 50.0 m ceiling (alias)."""

    def test_a_zero_translation_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(translation_m=0.0)
        assert payload["admits"] is True

    def test_a_translation_at_the_ceiling_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        # The neon runner uses `> 50.0` (strict), so exactly 50.0 admits.
        payload = g1_slam_relocalize_admits(translation_m=50.0)
        assert payload["admits"] is True

    def test_a_translation_above_the_ceiling_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(translation_m=50.1)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "translation_m"
        assert r["bound_key"] == "translation_max_m"
        assert r["comparison"] == "value > bound"

    def test_a_negative_translation_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        # A magnitude cannot be negative; the shape refusal has a
        # distinct bound_key so a caller reading the refusal knows
        # it is not the alias ceiling.
        payload = g1_slam_relocalize_admits(translation_m=-1.0)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "translation_m"
        assert r["bound_key"] == "translation_magnitude_floor"
        assert r["comparison"] == "value < bound"

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_translation_refuses(self, value: float) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(translation_m=value)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "translation_m"
        assert r["comparison"] == "non-finite"

    def test_a_boolean_translation_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(translation_m=True)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "translation_m"
        assert r["comparison"] == "non-int"


class TestTheRotationTraceDimensionIsGraded:
    """Rotation-trace clamp: 0.0 floor (sign / reflection), 3.0 ceiling (shape)."""

    def test_a_trace_at_the_floor_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        # The neon runner uses `< 0.0` (strict), so exactly 0.0 admits.
        payload = g1_slam_relocalize_admits(rotation_trace=0.0)
        assert payload["admits"] is True

    def test_a_trace_below_the_floor_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(rotation_trace=-0.5)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "rotation_trace"
        assert r["bound_key"] == "rotation_trace_min"
        assert r["comparison"] == "value < bound"

    def test_a_trace_at_the_ceiling_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(rotation_trace=3.0)
        assert payload["admits"] is True

    def test_a_trace_above_the_ceiling_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(rotation_trace=3.5)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "rotation_trace"
        assert r["bound_key"] == "rotation_trace_max"
        assert r["comparison"] == "value > bound"

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_trace_refuses(self, value: float) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(rotation_trace=value)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "rotation_trace"
        assert r["comparison"] == "non-finite"

    def test_a_boolean_trace_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(rotation_trace=False)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "rotation_trace"
        assert r["comparison"] == "non-int"


class TestAllThreeDimensionsAreGradedOnOneCall:
    """The neon runner short-circuits; this verb reports every violation.

    A caller planning a relocalise wants the whole shape of the
    refusal on one call so their next attempt fixes the whole
    triple.  A single-refusal payload would leave the caller
    re-planning against fitness alone and tripping on the
    translation ceiling the very next attempt.
    """

    def test_a_triple_with_two_violations_reports_both(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        # fitness below floor AND translation above ceiling: both
        # refusals must land on the refusals list.
        payload = g1_slam_relocalize_admits(fitness=0.1, translation_m=100.0)
        assert payload["admits"] is False
        dims = {r["dimension"] for r in payload["refusals"]}
        assert dims == {"fitness", "translation_m"}

    def test_a_triple_with_three_violations_reports_all_three(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(fitness=0.1, translation_m=100.0, rotation_trace=-0.5)
        assert payload["admits"] is False
        dims = {r["dimension"] for r in payload["refusals"]}
        assert dims == {"fitness", "translation_m", "rotation_trace"}

    def test_every_refusal_carries_the_module_local_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        # A rewording of the module-local text lands in one place;
        # this cell pins the invariant that every dimension's
        # refusal quotes the same shared text.
        payload = g1_slam_relocalize_admits(fitness=0.1, translation_m=100.0, rotation_trace=-0.5)
        texts = {r["text"] for r in payload["refusals"]}
        # All three dimensions share one text (a set of size 1),
        # and that text cites the umbrella issue.
        assert len(texts) == 1
        (text,) = texts
        assert "strands-labs/robots#358" in text


class TestEveryDeclaredRefusalIsReachable:
    """Rule-reachability grade: every advertised refusal is reached by an input.

    A refusal no input can reach is documentation for a case that
    never fires; the neon runner's `_try_relocalize` reads three
    shapes (fitness floor, translation ceiling, trace floor) and
    this verb widens them to two-sided clamps plus shape refusals.
    Every bound named in the envelope is reached by exactly one
    cell here, so a bound added to the descriptor without a
    reachable refusal fails this test.
    """

    def test_each_advertised_bound_key_is_reached_by_some_input(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        # Drive each refusal shape from a concrete input; the set
        # of bound_keys observed on the refusals matches the set
        # of bounds the envelope names.
        reached: set[str] = set()

        # Fitness floor and ceiling.
        for f in (0.1, 1.5):
            payload = g1_slam_relocalize_admits(fitness=f)
            for r in payload["refusals"]:
                reached.add(r["bound_key"])

        # Translation floor and ceiling.
        for t in (-1.0, 100.0):
            payload = g1_slam_relocalize_admits(translation_m=t)
            for r in payload["refusals"]:
                reached.add(r["bound_key"])

        # Rotation trace floor and ceiling.
        for rt in (-0.5, 3.5):
            payload = g1_slam_relocalize_admits(rotation_trace=rt)
            for r in payload["refusals"]:
                reached.add(r["bound_key"])

        # Every envelope bound is reached, plus the derived
        # translation_magnitude_floor shape refusal.
        assert reached == {
            "fitness_min",
            "fitness_max",
            "translation_magnitude_floor",
            "translation_max_m",
            "rotation_trace_min",
            "rotation_trace_max",
        }


class TestTheDecisionReadsNoFilesystemOrBusState:
    """A relocalise gate decision is a numeric one; no I/O runs here.

    Grades the module docstring's \"no bus is touched\" claim by
    verifying the two verbs produce byte-identical payloads for
    the same inputs called at two different points in the test
    run.  A verb that read the filesystem or a live bus would
    produce a different payload if the underlying state moved
    between the two reads.
    """

    def test_repeated_calls_produce_byte_identical_payloads(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_list_slam_relocalize_envelope,
            g1_slam_relocalize_admits,
        )

        list_a = g1_list_slam_relocalize_envelope()
        list_b = g1_list_slam_relocalize_envelope()
        assert list_a == list_b

        admits_a = g1_slam_relocalize_admits(fitness=0.5, translation_m=10.0, rotation_trace=1.0)
        admits_b = g1_slam_relocalize_admits(fitness=0.5, translation_m=10.0, rotation_trace=1.0)
        assert admits_a == admits_b


class TestTheRefusalTextIsAscii:
    """Whole-tree ASCII grader pins tool-result text; this cell pins the module's own."""

    def test_the_refusal_text_is_ascii(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_list_slam_relocalize_envelope,
        )

        payload = g1_list_slam_relocalize_envelope()
        # The refusal text lives on both verbs; grading it here
        # once suffices because the module's _REFUSAL_TEXT is a
        # single constant both verbs read.
        text = payload["refusals"][0]["text"]
        # No non-ASCII characters, no emoji, no combining marks.
        for ch in text:
            assert ord(ch) < 128, f"refusal text contains non-ASCII character U+{ord(ch):04X} ({ch!r})"

    def test_every_refusal_descriptor_text_is_ascii(self) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        # Sweep a triple that trips every dimension so every
        # dimension's refusal descriptor is graded on the same
        # invariant.
        payload = g1_slam_relocalize_admits(fitness=0.1, translation_m=100.0, rotation_trace=-0.5)
        for r in payload["refusals"]:
            text = r["text"]
            for ch in text:
                assert ord(ch) < 128, f"refusal text contains non-ASCII character U+{ord(ch):04X} ({ch!r})"


class TestAValidRealisticIcpResultAdmits:
    """A canonical good ICP fit -- indoor SLAM, ~2 m drift -- admits.

    Grades that the envelope does not accidentally refuse a
    legitimate relocalise the neon runner would admit.  The neon
    runner's own admit condition is fitness >= 0.3 AND translation
    magnitude <= 50 m AND trace >= 0; a realistic indoor SLAM fit
    reports fitness in [0.5, 0.9], translation in [0, 5] m, and
    trace close to 3.0 (small rotation correction).
    """

    @pytest.mark.parametrize(
        "fitness, translation_m, rotation_trace",
        [
            # Perfect identity fit (default).
            (1.0, 0.0, 3.0),
            # Realistic mid-quality indoor fit.
            (0.65, 1.5, 2.8),
            # Higher-drift fit still inside the ceiling.
            (0.45, 20.0, 2.0),
            # Boundary fit: exactly at the fitness floor.
            (0.3, 0.0, 3.0),
        ],
    )
    def test_a_realistic_indoor_fit_admits(self, fitness: float, translation_m: float, rotation_trace: float) -> None:
        from strands_robots.tools.g1.g1_slam_relocalize_envelope import (
            g1_slam_relocalize_admits,
        )

        payload = g1_slam_relocalize_admits(
            fitness=fitness,
            translation_m=translation_m,
            rotation_trace=rotation_trace,
        )
        assert payload["admits"] is True, (
            f"realistic ICP fit refused: fitness={fitness}, "
            f"translation_m={translation_m}, rotation_trace={rotation_trace}, "
            f"refusals={payload['refusals']}"
        )
