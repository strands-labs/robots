"""Tests for :mod:`strands_robots.tools.g1.g1_slam_voxel_dedup_envelope`.

The module ports the neon SLAM runner's ``_voxel_dedup`` authored
constant (``cagataycali/neon-the-g1/tools/g1_slam.py``) into a
read-only lookup pair.  The tests grade four things: import
hygiene (no optional SLAM stack loads at import), snapshot fidelity
(the envelope carries the neon-observed value verbatim), the
shape-decision matrix for the caller-supplied argument (bool, non-real,
non-finite, non-positive, past-float64), and refusal descriptor
invariants (ASCII, umbrella-issue citation, byte-identical repeated
calls).

The single refusal reason routes through
:func:`~strands_robots.utils.positive_finite_number_error`, so a
shape mistake is graded once by that shared domain and the verb
attaches the module-local :data:`_REFUSAL_TEXT` on top.

Refs strands-labs/robots#358.
"""

from __future__ import annotations

import importlib
import math
import sys

import pytest

MODULE_PATH = "strands_robots.tools.g1.g1_slam_voxel_dedup_envelope"


class TestTheImportPullsNoOptionalSlamModule:
    """The module docstring's import-hygiene contract, refs strands-labs/robots#358.

    A caller authoring a dedup plan before any SLAM extra is
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
            "voxel-dedup constant ports as a lookup, not as an SDK call"
        )

    def test_the_import_pulls_no_new_optional_slam_submodule(self) -> None:
        # numpy / open3d / kiss_icp are the SLAM extras the neon
        # runner's own ``_voxel_dedup`` reaches (numpy for the
        # np.floor/np.unique routing, no open3d/kiss_icp on this
        # pass).  numpy (top-level) may already be in sys.modules
        # from an earlier stdlib or test-fixture import, so the
        # cell checks the *delta*: a fresh submodule newly
        # imported by this module's load path is a rule violation;
        # an already-loaded numpy from an unrelated pathway is not.
        sys.modules.pop(MODULE_PATH, None)
        before = set(sys.modules)
        importlib.import_module(MODULE_PATH)
        added = set(sys.modules) - before
        leaked = {name for name in added if name.split(".")[0] in ("numpy", "open3d", "kiss_icp")}
        assert leaked == set(), (
            f"the import of {MODULE_PATH} newly pulled optional SLAM "
            f"submodules {sorted(leaked)}; the envelope is a "
            "module-level constant snapshot and reaches none of them. "
            "The np.floor / np.unique calls belong inside a future "
            "driver-side wrapper, refs strands-labs/robots#358."
        )


class TestTheEnvelopeSnapshotIsFaithful:
    """The envelope descriptor mirrors the neon runner's authored constant byte-for-byte."""

    def test_the_envelope_lists_the_neon_default(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_list_slam_voxel_dedup_envelope,
        )

        payload = g1_list_slam_voxel_dedup_envelope()
        assert payload["status"] == "success"
        envelope = payload["envelope"]
        # The envelope carries one field; a widen to the descriptor
        # (a second bound, an added quality hint) lands here.
        assert set(envelope) == {"voxel_dedup_neon_default_m"}
        # The value is the neon-runner-observed constant.
        assert envelope["voxel_dedup_neon_default_m"] == 0.05

    def test_the_envelope_carries_the_refusal_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_list_slam_voxel_dedup_envelope,
        )

        payload = g1_list_slam_voxel_dedup_envelope()
        refusals = payload["refusals"]
        assert len(refusals) == 1
        # The single refusal descriptor carries a text field
        # naming the surface (voxel dedup gate) and citing the
        # umbrella issue so a caller reading the refusal can find
        # the driver-side follow-up.
        text = refusals[0]["text"]
        assert "voxel dedup" in text
        assert "strands-labs/robots#358" in text

    def test_the_admits_verb_reports_the_same_envelope(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_list_slam_voxel_dedup_envelope,
            g1_slam_voxel_dedup_admits,
        )

        list_env = g1_list_slam_voxel_dedup_envelope()["envelope"]
        admits_env = g1_slam_voxel_dedup_admits()["envelope"]
        # A widen to the descriptor lands in one place because both
        # verbs read the same _envelope() helper.
        assert list_env == admits_env


class TestTheDefaultArgumentAdmits:
    """The verb's default argument lands on the admitted side of the shape check.

    The default is the neon-runner-observed value (5 cm), so a
    caller who does not pass an explicit argument lands on the
    boundary case the neon runner itself reads.
    """

    def test_the_default_argument_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        payload = g1_slam_voxel_dedup_admits()
        assert payload["status"] == "success"
        assert payload["admits"] is True
        assert payload["refusals"] == []

    def test_the_neon_observed_value_admits_explicitly(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        # Naming the explicit 0.05 argument grades the default's
        # value as well as the default's syntax.
        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=0.05)
        assert payload["admits"] is True


class TestTheShapeDimensionIsGraded:
    """Voxel edge shape: strict positive-finite float in the float64 range.

    Routed through the shared
    :func:`~strands_robots.utils.positive_finite_number_error`
    domain; every shape refusal the shared domain surfaces lands
    here on one dimension name.  The neon runner has no separate
    numeric clamp on the dedup voxel (it reads its authored
    constant inline), so the shared domain is the sole shape
    grade this verb reports.
    """

    @pytest.mark.parametrize(
        "value",
        [0.001, 0.02, 0.05, 0.1, 0.3, 1.0, 10.0, 1e100],
    )
    def test_a_positive_finite_float_admits(self, value: float) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=value)
        assert payload["admits"] is True, f"positive-finite float {value} refused: {payload['refusals']}"

    def test_a_zero_edge_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        # A dedup cell of zero collapses ``floor(pts * inf)`` to
        # non-finite; the shared domain refuses zero as a shape
        # violation.
        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=0.0)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "voxel_edge_m"
        assert r["comparison"] == "shared-domain"
        assert "> 0" in r["domain_error"]

    def test_a_negative_edge_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        # A negative cell size inverts the ``floor`` sign; the
        # shared domain refuses signed shape.
        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=-0.05)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "voxel_edge_m"
        assert r["comparison"] == "shared-domain"

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_edge_refuses(self, value: float) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=value)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "voxel_edge_m"
        assert r["comparison"] == "shared-domain"

    def test_a_boolean_edge_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        # ``True`` is ``int(1)`` which would otherwise silently
        # accept a 1.0 metre dedup cell (coarse enough to
        # collapse a working-area map to a single point). The
        # shared domain refuses bool as a shape violation.
        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=True)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "voxel_edge_m"
        assert r["comparison"] == "shared-domain"

    def test_a_false_edge_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        # ``False`` is ``int(0)`` and would otherwise be refused
        # by the ``<= 0`` half of the shared domain; the bool
        # shape check lands first so the refusal reads as a shape
        # mistake rather than a range one.
        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=False)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "voxel_edge_m"
        assert r["comparison"] == "shared-domain"

    def test_a_non_real_edge_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        # A caller passing a string (a JSON-round-tripped "0.05"
        # left un-decoded) reaches the shared domain's non-real
        # refusal.
        payload = g1_slam_voxel_dedup_admits(voxel_edge_m="0.05")  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "voxel_edge_m"
        assert r["comparison"] == "shared-domain"

    def test_a_value_past_the_float64_range_refuses(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        # ``10**400`` is positive and finite as a Python int but
        # past the float64 range; the shared domain refuses it
        # with a distinct reason so a caller sees the shape
        # mistake decidably.
        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=10**400)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["dimension"] == "voxel_edge_m"
        assert r["comparison"] == "shared-domain"
        assert "64-bit float" in r["domain_error"]


class TestEveryRefusalCarriesTheModuleLocalText:
    """Every refusal descriptor quotes the module-local :data:`_REFUSAL_TEXT`.

    A rewording of the module-local text lands in one place; this
    cell pins the invariant that every shape refusal quotes the
    same shared text alongside the shared-domain error message.
    """

    @pytest.mark.parametrize(
        "value",
        [0.0, -0.05, math.nan, math.inf, -math.inf, True, False, "0.05"],
    )
    def test_the_refusal_quotes_the_module_local_text(self, value: object) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=value)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert "strands-labs/robots#358" in r["text"]
        assert "voxel dedup" in r["text"]


class TestEveryDeclaredRefusalIsReachable:
    """Rule-reachability grade: every advertised refusal is reached by an input.

    The verb reports exactly one refusal shape (the shared-domain
    shape refusal); the envelope names one bound
    (``voxel_dedup_neon_default_m``).  This cell pins that the
    single advertised refusal cell is reached by at least one
    input.
    """

    def test_the_shared_domain_refusal_is_reached(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        # Drive the refusal shape from a concrete input; the
        # bound_key on the refusal matches the sole bound the
        # envelope names.
        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=0.0)
        assert payload["admits"] is False
        assert payload["refusals"][0]["bound_key"] == "voxel_dedup_neon_default_m"


class TestTheDecisionReadsNoFilesystemOrBusState:
    """A voxel-dedup shape decision is a numeric one; no I/O runs here.

    Grades the module docstring's "no bus is touched" claim by
    verifying the two verbs produce byte-identical payloads for
    the same inputs called at two different points in the test
    run.  A verb that read the filesystem or a live bus would
    produce a different payload if the underlying state moved
    between the two reads.
    """

    def test_repeated_calls_produce_byte_identical_payloads(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_list_slam_voxel_dedup_envelope,
            g1_slam_voxel_dedup_admits,
        )

        list_a = g1_list_slam_voxel_dedup_envelope()
        list_b = g1_list_slam_voxel_dedup_envelope()
        assert list_a == list_b

        admits_a = g1_slam_voxel_dedup_admits(voxel_edge_m=0.05)
        admits_b = g1_slam_voxel_dedup_admits(voxel_edge_m=0.05)
        assert admits_a == admits_b

        refused_a = g1_slam_voxel_dedup_admits(voxel_edge_m=0.0)
        refused_b = g1_slam_voxel_dedup_admits(voxel_edge_m=0.0)
        assert refused_a == refused_b


class TestTheRefusalTextIsAscii:
    """Whole-tree ASCII grader pins tool-result text; this cell pins the module's own."""

    def test_the_refusal_text_is_ascii(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_list_slam_voxel_dedup_envelope,
        )

        payload = g1_list_slam_voxel_dedup_envelope()
        # The refusal text lives on both verbs; grading it here
        # once suffices because the module's _REFUSAL_TEXT is a
        # single constant both verbs read.
        text = payload["refusals"][0]["text"]
        for ch in text:
            assert ord(ch) < 128, f"refusal text contains non-ASCII character U+{ord(ch):04X} ({ch!r})"

    def test_every_refusal_descriptor_text_is_ascii(self) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        # A refusing input surfaces the shape-refusal descriptor;
        # its text field carries the module-local refusal.  The
        # domain_error field is authored by the shared domain and
        # is not part of this module's ASCII surface.
        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=0.0)
        for r in payload["refusals"]:
            text = r["text"]
            for ch in text:
                assert ord(ch) < 128, f"refusal text contains non-ASCII character U+{ord(ch):04X} ({ch!r})"


class TestARealisticProposalAdmits:
    """A canonical dedup voxel edge -- 5 cm indoor, 30 cm outdoor -- admits.

    Grades that the shared-domain shape refusal does not
    accidentally refuse a legitimate proposal a caller planning
    an indoor or outdoor SLAM run would offer.  The neon runner's
    own authored value is 5 cm; a wider outdoor proposal like
    30 cm and a finer millimetre-scale proposal like 1 cm both
    sit inside the shared positive-finite domain.
    """

    @pytest.mark.parametrize(
        "value",
        [
            # Neon-observed (indoor, ~50k returns per frame).
            0.05,
            # Finer indoor proposal (millimetre-scale scan).
            0.01,
            # Coarser outdoor proposal (wide-area map).
            0.30,
            # Very coarse outdoor proposal (km-scale exploration).
            1.00,
        ],
    )
    def test_a_realistic_edge_admits(self, value: float) -> None:
        from strands_robots.tools.g1.g1_slam_voxel_dedup_envelope import (
            g1_slam_voxel_dedup_admits,
        )

        payload = g1_slam_voxel_dedup_admits(voxel_edge_m=value)
        assert payload["admits"] is True, (
            f"realistic dedup edge refused: voxel_edge_m={value}, refusals={payload['refusals']}"
        )
