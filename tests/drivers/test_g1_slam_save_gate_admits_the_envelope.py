"""Tests for :mod:`strands_robots.tools.g1.g1_slam_save_envelope`.

The module ports the neon SLAM runner's ``save_map`` gate
(``cagataycali/neon-the-g1/tools/g1_slam.py``) into a read-only lookup
pair.  The tests grade three things: import hygiene (no optional SLAM
stack loads at import), snapshot fidelity (the envelope carries the
neon-observed constant on both verbs), and the admit/refuse decision
matrix for the two graded dimensions (chunks_count, name).

The refusals share one module-local :data:`_REFUSAL_TEXT`, so a misread
of any one dimension is graded by both the dimension-specific cell and
by a cross-cell that a multi-dimension violation reports every refusal
at once -- the neon runner short-circuits on the first refusal, but a
caller planning a save wants every violated rule named so the next
attempt fixes the whole pair.

Twin of :mod:`~strands_robots.tools.g1.g1_slam_map_liveness_envelope`
(strands-labs/robots#3011) -- that envelope names the *load*-side
floor and this envelope names the *save*-side floor on the same
accumulated chunks list, refs strands-labs/robots#358.
"""

from __future__ import annotations

import importlib
import sys

import pytest

MODULE_PATH = "strands_robots.tools.g1.g1_slam_save_envelope"


class TestTheImportPullsNoOptionalSlamModule:
    """The module docstring's import-hygiene contract, refs strands-labs/robots#358.

    A caller authoring a save plan before any SLAM extra is installed
    on their host still gets the envelope back verbatim; the module's
    advertised no-optional-import property is asserted against the
    process's own :data:`sys.modules` after import.
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
            "save gate ports as a lookup, not as an SDK call"
        )

    def test_the_import_pulls_no_new_optional_slam_submodule(self) -> None:
        # numpy / open3d / kiss_icp are the SLAM extra the neon
        # runner's own save_map reaches (numpy.savez).  numpy
        # (top-level) may already be in sys.modules from an earlier
        # stdlib or test-fixture import, so the cell checks the
        # *delta*: a fresh submodule newly imported by this module's
        # load path is a rule violation; an already-loaded numpy
        # from an unrelated pathway is not.  Pinned so a future
        # edit that reaches into e.g. numpy.savez for a compile-time
        # write-size ceiling fails this cell first.
        sys.modules.pop(MODULE_PATH, None)
        before = set(sys.modules)
        importlib.import_module(MODULE_PATH)
        added = set(sys.modules) - before
        leaked = {name for name in added if name.split(".")[0] in ("numpy", "open3d", "kiss_icp")}
        assert leaked == set(), (
            f"the import of {MODULE_PATH} newly pulled optional SLAM "
            f"submodules {sorted(leaked)}; the envelope is a "
            "module-level constant snapshot and reaches none of them. "
            "The numpy.savez / open3d / kiss_icp calls belong inside a "
            "future driver-side wrapper, refs strands-labs/robots#358."
        )


class TestTheEnvelopeSnapshotIsFaithful:
    """The envelope descriptor mirrors the neon runner's own bound byte-for-byte."""

    def test_the_envelope_lists_the_neon_runner_constant(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_list_slam_save_envelope,
        )

        payload = g1_list_slam_save_envelope()
        assert payload["status"] == "success"
        envelope = payload["envelope"]
        # Every field the module docstring names is present on the
        # envelope; a widen to the descriptor lands in one place.
        assert set(envelope) == {"chunks_count_min"}
        # The value is the neon-runner-observed constant.
        assert envelope["chunks_count_min"] == 1

    def test_the_floor_is_a_positive_int(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_list_slam_save_envelope,
        )

        floor = g1_list_slam_save_envelope()["envelope"]["chunks_count_min"]
        # A count, not a fraction (the neon runner reads ``not chunks``,
        # which is a strict emptiness test on a list); pinned as an
        # int rather than a float so a future edit that switches the
        # constant to 1.0 fails this cell first.
        assert isinstance(floor, int)
        assert not isinstance(floor, bool)
        assert floor >= 1

    def test_the_envelope_carries_the_refusal_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_list_slam_save_envelope,
        )

        payload = g1_list_slam_save_envelope()
        refusals = payload["refusals"]
        assert len(refusals) == 1
        text = refusals[0]["text"]
        assert "save" in text
        assert "strands-labs/robots#358" in text

    def test_the_admits_verb_reports_the_same_envelope(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_list_slam_save_envelope,
            g1_slam_save_admits,
        )

        list_env = g1_list_slam_save_envelope()["envelope"]
        admits_env = g1_slam_save_admits()["envelope"]
        # A widen to the descriptor lands in one place because both
        # verbs read the same _envelope() helper.
        assert list_env == admits_env


class TestTheDefaultPairAdmits:
    """The verb's default arguments land on the admitted side of every rule."""

    def test_the_default_pair_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        payload = g1_slam_save_admits()
        assert payload["status"] == "success"
        assert payload["admits"] is True
        assert payload["refusals"] == []


class TestTheChunksCountDimensionIsGraded:
    """Chunks-count clamp: floor at 1 inclusive (the neon runner's ``not chunks`` check)."""

    def test_chunks_count_at_the_floor_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        payload = g1_slam_save_admits(chunks_count=1)
        assert payload["admits"] is True

    def test_chunks_count_well_above_the_floor_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        # A realistic accumulated map might have thousands of
        # chunks; well above the floor stays inside the envelope.
        payload = g1_slam_save_admits(chunks_count=5000)
        assert payload["admits"] is True

    def test_chunks_count_of_zero_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        payload = g1_slam_save_admits(chunks_count=0)
        assert payload["admits"] is False
        assert len(payload["refusals"]) == 1
        r = payload["refusals"][0]
        assert r["dimension"] == "chunks_count"
        assert r["bound_key"] == "chunks_count_min"
        assert r["bound"] == 1
        assert r["comparison"] == "value < bound"
        assert "strands-labs/robots#358" in r["text"]

    def test_negative_chunks_count_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        payload = g1_slam_save_admits(chunks_count=-1)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "chunks_count"
        assert r["comparison"] == "value < bound"

    @pytest.mark.parametrize("value", [True, False])
    def test_boolean_chunks_count_refuses(self, value: bool) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        # ``True`` is ``int(1)`` and would otherwise silently look
        # up 1 (a legitimate minimum accumulation).  Naming the
        # refusal at the boundary surfaces the type mistake.
        payload = g1_slam_save_admits(chunks_count=value)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "chunks_count"
        assert r["comparison"] == "non-int"

    @pytest.mark.parametrize("value", [1.0, 1.5, "1", None, [1]])
    def test_non_int_chunks_count_refuses(self, value: object) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        payload = g1_slam_save_admits(chunks_count=value)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "chunks_count"
        assert r["comparison"] == "non-int"


class TestTheNameDimensionIsGraded:
    """Name-shape rules: str, non-empty, no path separators, no traversal."""

    @pytest.mark.parametrize("name", ["map", "session_1", "office-1", "a", "map_2026_08_30"])
    def test_a_plain_alphanumeric_stem_admits(self, name: str) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        payload = g1_slam_save_admits(name=name)
        assert payload["admits"] is True, f"plain stem {name!r} was refused; refusals={payload['refusals']}"

    def test_empty_name_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        payload = g1_slam_save_admits(name="")
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "name"
        assert r["bound_key"] == "name_shape"
        assert r["comparison"] == "empty"

    @pytest.mark.parametrize(
        "name",
        [
            "../etc/passwd",
            "..",
            "sub/dir/map",
            "map\\path",
            "/abs/path/map",
            "./hidden",
            ".hidden",
        ],
    )
    def test_path_shape_name_refuses(self, name: str) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        payload = g1_slam_save_admits(name=name)
        assert payload["admits"] is False, f"path-shape name {name!r} was admitted; expected refusal"
        r = payload["refusals"][0]
        assert r["dimension"] == "name"
        assert r["comparison"] == "path-shape"

    @pytest.mark.parametrize("value", [True, False])
    def test_boolean_name_refuses(self, value: bool) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        # ``True`` is ``int(1)``, not a ``str``; the ``isinstance
        # (name, str)`` branch already excludes it, but naming a
        # ``bool``-specific comparison surfaces the type mistake
        # rather than reporting it as a generic non-str.
        payload = g1_slam_save_admits(name=value)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "name"
        assert r["comparison"] == "non-int"

    @pytest.mark.parametrize("value", [None, 42, 3.14, ["map"], {"name": "map"}])
    def test_non_str_name_refuses(self, value: object) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        payload = g1_slam_save_admits(name=value)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "name"
        assert r["comparison"] == "non-str"


class TestBothDimensionsAreGradedOnOneCall:
    """The neon runner short-circuits; this verb reports every violation."""

    def test_a_pair_with_two_violations_reports_both(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        # chunks_count below floor AND name path-shape: both
        # refusals must land on the refusals list.
        payload = g1_slam_save_admits(chunks_count=0, name="../etc/passwd")
        assert payload["admits"] is False
        dims = {r["dimension"] for r in payload["refusals"]}
        assert dims == {"chunks_count", "name"}

    def test_every_refusal_carries_the_module_local_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        # A rewording of the module-local text lands in one place;
        # this cell pins the invariant that every dimension's
        # refusal quotes the same shared text.
        payload = g1_slam_save_admits(chunks_count=0, name="../etc/passwd")
        texts = {r["text"] for r in payload["refusals"]}
        assert len(texts) == 1
        (text,) = texts
        assert "strands-labs/robots#358" in text


class TestEveryDeclaredRefusalIsReachable:
    """Rule-reachability grade: every advertised refusal is reached by an input.

    A refusal no input can reach is documentation for a case that
    never fires; every rule this verb names is reached by exactly
    one cell here, so a bound added without a reachable refusal
    fails this test.
    """

    def test_each_advertised_bound_key_is_reached_by_some_input(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        reached: set[str] = set()

        # chunks_count floor.
        for cc in (0, -5):
            for r in g1_slam_save_admits(chunks_count=cc)["refusals"]:
                reached.add(r["bound_key"])

        # name shape rules (empty, path-shape, non-str, bool all
        # collapse to the single ``name_shape`` bound_key with
        # different ``comparison`` strings, so one input per
        # comparison suffices).
        for nm in ("", "../etc/passwd", 42):
            for r in g1_slam_save_admits(name=nm)["refusals"]:  # type: ignore[arg-type]
                reached.add(r["bound_key"])

        # Every envelope rule is reached.
        assert reached == {"chunks_count_min", "name_shape"}


class TestTheDecisionReadsNoFilesystemOrBusState:
    """A save-gate decision is a rule-based one; no I/O runs here.

    Grades the module docstring's "no bus is touched" claim by
    verifying the two verbs produce byte-identical payloads for the
    same inputs called at two different points in the test run.  A
    verb that read the filesystem or a live bus would produce a
    different payload if the underlying state moved between the two
    reads.
    """

    def test_repeated_calls_produce_byte_identical_payloads(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_list_slam_save_envelope,
            g1_slam_save_admits,
        )

        list_a = g1_list_slam_save_envelope()
        list_b = g1_list_slam_save_envelope()
        assert list_a == list_b

        admits_a = g1_slam_save_admits(chunks_count=100, name="office")
        admits_b = g1_slam_save_admits(chunks_count=100, name="office")
        assert admits_a == admits_b


class TestTheRefusalTextIsAscii:
    """Whole-tree ASCII grader pins tool-result text; this cell pins the module's own."""

    def test_the_refusal_text_is_ascii(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_list_slam_save_envelope,
        )

        payload = g1_list_slam_save_envelope()
        text = payload["refusals"][0]["text"]
        for ch in text:
            assert ord(ch) < 128, f"refusal text contains non-ASCII character U+{ord(ch):04X} ({ch!r})"

    def test_every_refusal_descriptor_text_is_ascii(self) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        # Sweep a pair that trips every rule so every rule's
        # refusal descriptor is graded on the same invariant.
        payload = g1_slam_save_admits(chunks_count=0, name="../etc/passwd")
        for r in payload["refusals"]:
            text = r["text"]
            for ch in text:
                assert ord(ch) < 128, f"refusal text contains non-ASCII character U+{ord(ch):04X} ({ch!r})"


class TestARealisticSavePairAdmits:
    """A canonical good save -- indoor SLAM, a few thousand chunks -- admits."""

    @pytest.mark.parametrize(
        "chunks_count, name",
        [
            # Boundary save: exactly one chunk (the neon-observed
            # minimum accumulation).
            (1, "map"),
            # Realistic indoor SLAM after a room-scale scan.
            (200, "office"),
            # Long session with periodic compaction.
            (5000, "warehouse_2026_08_30"),
            # Hyphenated stem (common shell naming).
            (500, "session-1"),
        ],
    )
    def test_a_realistic_save_admits(self, chunks_count: int, name: str) -> None:
        from strands_robots.tools.g1.g1_slam_save_envelope import (
            g1_slam_save_admits,
        )

        payload = g1_slam_save_admits(chunks_count=chunks_count, name=name)
        assert payload["admits"] is True, (
            f"realistic save refused: chunks_count={chunks_count}, name={name!r}, refusals={payload['refusals']}"
        )
