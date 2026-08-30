"""Tests for :mod:`strands_robots.tools.g1.g1_slam_map_liveness_envelope`.

The module ports the neon SLAM runner's ``_try_relocalize``
precondition (``cagataycali/neon-the-g1/tools/g1_slam.py``) into a
read-only lookup pair.  The tests grade three things: import
hygiene (no optional SLAM stack loads at import), snapshot
fidelity (the envelope carries the neon-observed ``100``
point-count floor on both verbs), and the admit/refuse decision
matrix for the point-count dimension.

The single refusal uses one module-local :data:`_REFUSAL_TEXT` on
both a below-floor rejection and a shared-domain shape mistake, so
a misread of either grade surfaces the same remedy string on the
same surface -- consistent with the twin envelope
:mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope` (the
merged strands-labs/robots#3006).

Refs strands-labs/robots#358.
"""

from __future__ import annotations

import importlib
import sys

import pytest

MODULE_PATH = "strands_robots.tools.g1.g1_slam_map_liveness_envelope"


class TestTheImportPullsNoOptionalSlamModule:
    """The module docstring's import-hygiene contract, refs strands-labs/robots#358.

    A caller authoring a relocalise plan before any SLAM extra is
    installed on their host still gets the floor back verbatim;
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
            "precondition ports as a lookup, not as an SDK call"
        )

    def test_the_import_pulls_no_new_optional_slam_submodule(self) -> None:
        # numpy / open3d / kiss_icp are the SLAM extra the neon
        # runner's own _try_relocalize reaches.  numpy (top-level)
        # is often pre-loaded by the test session; the assertion
        # here is that *this* import does not pull any of them, so
        # a caller who imports the envelope on a host without the
        # SLAM extra still lands on a working module.
        sys.modules.pop(MODULE_PATH, None)
        before = set(sys.modules)
        importlib.import_module(MODULE_PATH)
        added = set(sys.modules) - before
        leaked = {
            name
            for name in added
            if name == "open3d" or name.startswith("open3d.") or name == "kiss_icp" or name.startswith("kiss_icp.")
        }
        assert leaked == set(), (
            f"the import of {MODULE_PATH} pulled SLAM-extra "
            f"submodules {sorted(leaked)}; the module must load "
            "verbatim on a host without the SLAM extra"
        )


class TestTheEnvelopeSnapshotFidelity:
    """The floor matches the neon runner's observed constant, refs strands-labs/robots#358."""

    def test_the_floor_is_a_positive_int(self) -> None:
        from strands_robots.tools.g1 import g1_slam_map_liveness_envelope as m

        # The floor is a discrete point count; it must be a strict
        # int (not a numpy int64, not a bool) so the shared
        # positive_count_error validator admits it on the default
        # path.  The runner reads len(map_pts), which is Python's
        # own int type, so this pins the type on both sides.
        assert isinstance(m._MAP_LIVENESS_MIN, int)
        assert not isinstance(m._MAP_LIVENESS_MIN, bool)
        assert m._MAP_LIVENESS_MIN > 0

    def test_the_neon_floor_matches_the_observed_snapshot(self) -> None:
        from strands_robots.tools.g1 import g1_slam_map_liveness_envelope as m

        # The neon runner reads `len(map_pts) < 100` and refuses on
        # strict-less; the envelope names 100 as the inclusive
        # lower bound.  A widen to this constant is a runner-side
        # change that this test would catch as a diverged
        # snapshot.
        assert m._MAP_LIVENESS_MIN == 100

    def test_g1_list_slam_map_liveness_envelope_returns_the_full_envelope(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            g1_list_slam_map_liveness_envelope,
        )

        payload = g1_list_slam_map_liveness_envelope()
        assert payload["status"] == "success"
        assert payload["envelope"] == {"point_count_min": 100}
        # Exactly one refusal descriptor -- the module-local text
        # a future write verb would surface on a below-floor map.
        assert len(payload["refusals"]) == 1
        text = payload["refusals"][0]["text"]
        assert "map liveness gate refused" in text
        assert "strands-labs/robots#358" in text

    def test_the_admits_envelope_matches_the_list_envelope(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            g1_list_slam_map_liveness_envelope,
            g1_slam_map_liveness_admits,
        )

        # The admits payload names the same envelope as the list
        # payload, so a caller who read the envelope from admits
        # (on a rejected count) reads the same fields as a caller
        # who read it from the dedicated list verb.  Guards
        # against a widen that landed in one verb only.
        list_env = g1_list_slam_map_liveness_envelope()["envelope"]
        admits_env = g1_slam_map_liveness_admits(point_count=100)["envelope"]
        assert list_env == admits_env


class TestG1SlamMapLivenessAdmitsAtTheFloor:
    """Boundary case: exactly the floor admits, refs strands-labs/robots#358.

    The neon runner's own check reads `len(map_pts) < 100` and
    refuses on strict-less, so the boundary case at exactly 100 is
    the case the runner admits.  This test pins that boundary in
    the envelope's own admits verb.
    """

    def test_g1_slam_map_liveness_admits_at_the_min_boundary(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            g1_slam_map_liveness_admits,
        )

        payload = g1_slam_map_liveness_admits(point_count=100)
        assert payload["status"] == "success"
        assert payload["admits"] is True
        assert payload["refusals"] == []

    def test_g1_slam_map_liveness_admits_above_the_min_boundary(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            g1_slam_map_liveness_admits,
        )

        # A well-populated map admits without qualification; the
        # runner would proceed to the match-quality dimensions the
        # twin envelope names.
        payload = g1_slam_map_liveness_admits(point_count=50_000)
        assert payload["admits"] is True
        assert payload["refusals"] == []

    def test_g1_slam_map_liveness_admits_default_call_is_the_min_boundary(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            g1_slam_map_liveness_admits,
        )

        # The default argument is the observed floor; a zero-arg
        # call must land on the admitted side so a caller probing
        # the envelope shape reads a clean admits payload.
        payload = g1_slam_map_liveness_admits()
        assert payload["admits"] is True
        assert payload["refusals"] == []


class TestG1SlamMapLivenessRefusesBelowTheMin:
    """A point count strictly below the floor refuses on the runner's own comparison."""

    def test_g1_slam_map_liveness_admits_below_the_min(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            _REFUSAL_TEXT,
            g1_slam_map_liveness_admits,
        )

        # 99 is the boundary-below case: the runner reads
        # len(map_pts) < 100 and refuses on strict-less, so 99
        # refuses.  The refusal names the dimension, the offending
        # value, the clamp it violated, the "value < bound"
        # comparison, and the module-local text.
        payload = g1_slam_map_liveness_admits(point_count=99)
        assert payload["admits"] is False
        assert len(payload["refusals"]) == 1
        r = payload["refusals"][0]
        assert r["dimension"] == "point_count"
        assert r["value"] == 99
        assert r["bound_key"] == "point_count_min"
        assert r["bound"] == 100
        assert r["comparison"] == "value < bound"
        assert r["text"] == _REFUSAL_TEXT

    def test_g1_slam_map_liveness_admits_at_one_refuses_on_min(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            g1_slam_map_liveness_admits,
        )

        # A single point admits the shared positive_count_error
        # domain (a positive int) but refuses on the map-liveness
        # floor -- the shape is a valid count but the runner's
        # ICP would still refuse for lack of correspondence.  The
        # refusal names the "value < bound" comparison rather
        # than the shared-domain shape refusal.
        payload = g1_slam_map_liveness_admits(point_count=1)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "value < bound"


class TestG1SlamMapLivenessRefusesSharedDomainShapeMistakes:
    """The shared positive_count_error refuses bool, non-int, and value < 1."""

    def test_g1_slam_map_liveness_admits_refuses_zero_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            g1_slam_map_liveness_admits,
        )

        # 0 is not a positive count; the shared domain refuses on
        # the shape rather than the map-liveness floor.  This
        # mirrors the runner's own precondition treating an empty
        # array under the `map_pts is None` half (a distinct
        # remedy from "build a bigger map").
        payload = g1_slam_map_liveness_admits(point_count=0)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"
        assert "domain_error" in r
        assert "positive integer" in r["domain_error"]

    def test_g1_slam_map_liveness_admits_refuses_negative_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            g1_slam_map_liveness_admits,
        )

        payload = g1_slam_map_liveness_admits(point_count=-1)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"

    def test_g1_slam_map_liveness_admits_refuses_bool_true_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            g1_slam_map_liveness_admits,
        )

        # bool is an int subclass whose True would otherwise land
        # as a silent count of 1; the shared domain refuses it
        # explicitly so the shape mistake is decidable.
        payload = g1_slam_map_liveness_admits(point_count=True)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"

    def test_g1_slam_map_liveness_admits_refuses_bool_false_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            g1_slam_map_liveness_admits,
        )

        payload = g1_slam_map_liveness_admits(point_count=False)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"

    @pytest.mark.parametrize("bad", [1.0, 100.0, "100", None])
    def test_g1_slam_map_liveness_admits_refuses_non_int_as_shape_mistake(self, bad: object) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            g1_slam_map_liveness_admits,
        )

        # A caller who computed a float count (a percentage, an
        # np.float64 read from a stat) or handed a str/None must
        # see the shape refusal, not the map-liveness floor
        # refusal -- the shared domain names the type mistake
        # decidably before the floor is asked.
        payload = g1_slam_map_liveness_admits(point_count=bad)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"


class TestTheRefusalTextIsShared:
    """A misread of any grade surfaces the same remedy string, refs strands-labs/robots#358."""

    def test_the_below_floor_refusal_uses_the_module_local_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            _REFUSAL_TEXT,
            g1_slam_map_liveness_admits,
        )

        payload = g1_slam_map_liveness_admits(point_count=50)
        assert payload["refusals"][0]["text"] == _REFUSAL_TEXT

    def test_the_shared_domain_refusal_uses_the_module_local_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            _REFUSAL_TEXT,
            g1_slam_map_liveness_admits,
        )

        payload = g1_slam_map_liveness_admits(point_count=0)
        assert payload["refusals"][0]["text"] == _REFUSAL_TEXT

    def test_the_list_verb_uses_the_module_local_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_liveness_envelope import (
            _REFUSAL_TEXT,
            g1_list_slam_map_liveness_envelope,
        )

        payload = g1_list_slam_map_liveness_envelope()
        assert payload["refusals"][0]["text"] == _REFUSAL_TEXT
