"""The shared bridge base fails a mis-secured inbound command surface at build time.

An inbound ``joint_command`` subscription lets any participant on the DDS domain
drive a physical arm, so :class:`strands_robots.ros_telemetry.RosTelemetryBase`
validates the drivable surface at construction rather than silently degrading:

* ``joint_limits`` must be a well-formed ``{"<motor>.pos": (min, max)}`` mapping,
  or the
  bridge refuses to build (a malformed bound can never become a silent mid-run
  rejection of every command).
* An enabled command surface requires DDS Security credentials *or* an explicit
  operator opt-out; neither present is a hard refusal, not a warning-and-continue.

These contracts are transport-agnostic, so they are exercised directly against
the base without rclpy or cyclonedds.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.ros_telemetry import (
    ROS2_INSECURE_ENV,
    RosTelemetryBase,
)
from strands_robots.utils import finite_number_error


class _Msg:
    def __init__(self, name: list[str], position: list[float]) -> None:
        self.name = name
        self.position = position


# --- joint-limit validation (construction-time, fail fast) -------------------


def test_valid_joint_limits_are_normalized_to_float_pairs() -> None:
    out = RosTelemetryBase._validate_joint_limits({"shoulder": (0, 10), "elbow": [-1.5, 1.5]})
    assert out == {"shoulder": (0.0, 10.0), "elbow": (-1.5, 1.5)}
    # Values are floats regardless of the input numeric type.
    assert all(isinstance(v, float) for pair in out.values() for v in pair)


def test_absent_joint_limits_normalize_to_none() -> None:
    assert RosTelemetryBase._validate_joint_limits(None) is None


def test_non_dict_joint_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be a dict"):
        RosTelemetryBase._validate_joint_limits([("a", 0, 1)])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        {"a": (0.0,)},  # too few values to unpack
        {"a": (0.0, 1.0, 2.0)},  # too many values to unpack
        {"a": 5.0},  # not a pair at all
    ],
)
def test_malformed_bound_pairs_are_rejected_naming_the_joint(bad: dict[str, object]) -> None:
    """A value that is not a two-element pair is reported on its shape.

    A pair whose *elements* are unusable (non-numeric, non-finite, past the
    float64 range) is reported per side instead - see
    :class:`TestANonFiniteBoundIsRefusedAtConstruction`, which names ``min`` or
    ``max`` rather than the pair as a whole.
    """
    with pytest.raises(ValueError, match=r"joint_limits\['a'\] must be a \(min, max\) numeric pair"):
        RosTelemetryBase._validate_joint_limits(bad)  # type: ignore[arg-type]


def test_inverted_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="min 2.0 > max 1.0"):
        RosTelemetryBase._validate_joint_limits({"a": (2.0, 1.0)})


# --- DDS security-config validation ------------------------------------------


def test_complete_dds_security_config_is_returned_unchanged() -> None:
    cfg = {
        "identity_ca": "file:/ca.pem",
        "certificate": "file:/cert.pem",
        "private_key": "file:/key.pem",
        "governance": "file:/gov.p7s",
        "permissions": "file:/perm.p7s",
    }
    assert RosTelemetryBase._validate_dds_security_config(cfg) is cfg


def test_non_dict_dds_security_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a dict"):
        RosTelemetryBase._validate_dds_security_config("file:/all.pem")


@pytest.mark.parametrize("missing_key", ["identity_ca", "certificate", "private_key", "governance", "permissions"])
def test_dds_security_config_missing_any_required_key_is_rejected(missing_key: str) -> None:
    cfg = {
        "identity_ca": "file:/ca.pem",
        "certificate": "file:/cert.pem",
        "private_key": "file:/key.pem",
        "governance": "file:/gov.p7s",
        "permissions": "file:/perm.p7s",
    }
    del cfg[missing_key]
    with pytest.raises(ValueError, match=r"missing required keys"):
        RosTelemetryBase._validate_dds_security_config(cfg)


def test_dds_security_config_empty_credential_counts_as_missing() -> None:
    cfg = {
        "identity_ca": "file:/ca.pem",
        "certificate": "   ",  # whitespace-only is not a credential
        "private_key": "file:/key.pem",
        "governance": "file:/gov.p7s",
        "permissions": "file:/perm.p7s",
    }
    with pytest.raises(ValueError, match="certificate"):
        RosTelemetryBase._validate_dds_security_config(cfg)


# --- inbound command-surface security gate -----------------------------------


def test_telemetry_only_bridge_is_never_gated() -> None:
    # A publish-only bridge exposes no drivable surface, so it needs no security
    # config and no opt-out even on a bare DDS graph.
    RosTelemetryBase._require_secure_command_surface(enable_commands=False, dds_security_config=None)


def test_secured_command_surface_is_allowed() -> None:
    RosTelemetryBase._require_secure_command_surface(
        enable_commands=True,
        dds_security_config={"identity_ca": "file:/ca.pem"},
    )


def test_enabled_command_surface_without_security_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ROS2_INSECURE_ENV, raising=False)
    with pytest.raises(ValueError, match="unsecured DDS graph"):
        RosTelemetryBase._require_secure_command_surface(enable_commands=True, dds_security_config=None)


def test_explicit_operator_opt_out_permits_unsecured_command_surface(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ROS2_INSECURE_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="strands_robots.ros_telemetry"):
        RosTelemetryBase._require_secure_command_surface(enable_commands=True, dds_security_config=None)
    # The opt-out is loud: an operator override must leave an audit trail.
    assert any("UNSECURED DDS graph" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "value,accepted",
    [("1", True), ("true", True), ("YES", True), ("0", False), ("", False), ("maybe", False)],
)
def test_insecure_opt_out_truthiness_contract(monkeypatch: pytest.MonkeyPatch, value: str, accepted: bool) -> None:
    monkeypatch.setenv(ROS2_INSECURE_ENV, value)
    assert RosTelemetryBase._insecure_opt_out() is accepted


# --- inbound command dispatch: rejected send_action is surfaced, not raised ---


def test_drive_from_command_surfaces_error_status_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    base = RosTelemetryBase()

    class _RejectingRobot:
        def send_action(self, action: dict[str, float]) -> dict[str, object]:
            return {"status": "error", "content": [{"text": "joint out of range"}]}

    with caplog.at_level(logging.WARNING, logger="strands_robots.ros_telemetry"):
        base._drive_from_command(_RejectingRobot(), _Msg(["a"], [0.5]))  # must not raise
    assert any("rejected joint_command" in r.getMessage() for r in caplog.records)


def test_out_of_range_command_is_dropped_whole(caplog: pytest.LogCaptureFixture) -> None:
    # One joint outside its declared bound rejects the ENTIRE command; no partial
    # application can drive part of the arm to a surprising pose.
    base = RosTelemetryBase()
    limits = {"a": (-1.0, 1.0), "b": (-1.0, 1.0)}
    with caplog.at_level(logging.WARNING, logger="strands_robots.ros_telemetry"):
        action = base._command_action(_Msg(["a", "b"], [0.5, 2.0]), joint_limits=limits)
    assert action is None
    # An unbounded joint alongside bounded ones is accepted when all are in range.
    assert base._command_action(_Msg(["a", "c"], [0.5, 99.0]), joint_limits=limits) == {"a": 0.5, "c": 99.0}


# --- a non-finite bound the ordering check cannot see ------------------------


_NAN = float("nan")
_INF = float("inf")


class TestANonFiniteBoundIsRefusedAtConstruction:
    """A bound that cannot bound anything refuses the bridge, not every command.

    ``_validate_joint_limits`` exists so a malformed bound surfaces at
    construction rather than as a silent mid-run rejection. Its ordering check
    (``low > high``) cannot see a non-finite bound, because every comparison
    against ``nan`` is ``False`` and ``inf > inf`` is ``False`` too. Such a pair
    used to be accepted, and then the ``low <= pos <= high`` test in
    :meth:`~strands_robots.ros_telemetry.RosTelemetryBase._command_action` was
    ``False`` for every position - so the bridge came up clean and dropped every
    inbound ``joint_command`` for that joint, which is exactly the failure the
    validator was written to prevent.

    Each bound now goes through :func:`strands_robots.utils.finite_number_error`,
    the shared domain for a signed physical quantity, so the wording matches
    every other caller and cannot drift. The sibling declaration of this same
    parameter,
    :class:`~strands_robots.simulation.isaac.delta_eef.IsaacDeltaEEFController`,
    already refused a non-finite bound for the same reason.
    """

    def test_the_ordering_check_cannot_see_a_non_finite_bound(self) -> None:
        """The premise: why ``low > high`` admits a bound that admits nothing."""
        assert (1.9 > _NAN) is False
        assert (_NAN > _NAN) is False
        assert (_INF > _INF) is False
        assert (-_INF > -_INF) is False
        # ... and the range test in _command_action is False for every position,
        # which is how an accepted bound became a rejection of every command.
        assert (-1.9 <= 0.5 <= _NAN) is False
        assert (_INF <= 0.5 <= _INF) is False

    @pytest.mark.parametrize("side", ["min", "max"])
    @pytest.mark.parametrize("bound", [_NAN, _INF, -_INF], ids=["nan", "+inf", "-inf"])
    def test_a_non_finite_bound_on_either_side_is_refused(self, bound: float, side: str) -> None:
        bounds = (bound, 1.9) if side == "min" else (-1.9, bound)
        with pytest.raises(ValueError, match=rf"joint_limits\['shoulder_pan'\]: {side} must be a finite number"):
            RosTelemetryBase._validate_joint_limits({"shoulder_pan": bounds})

    @pytest.mark.parametrize(
        "bounds",
        [(_NAN, _NAN), (_INF, _INF), (-_INF, -_INF)],
        ids=["nan-nan", "inf-inf", "neginf-neginf"],
    )
    def test_a_pair_that_passes_the_ordering_check_is_still_refused(self, bounds: tuple[float, float]) -> None:
        """These three are the pairs ``low > high`` is blind to on both sides."""
        assert (bounds[0] > bounds[1]) is False, "fixture no longer passes the ordering check"
        with pytest.raises(ValueError, match="must be a finite number"):
            RosTelemetryBase._validate_joint_limits({"j": bounds})

    def test_the_refusal_is_the_shared_domain_verdict_verbatim(self) -> None:
        """One wording for this domain, so it cannot drift from its other callers."""
        with pytest.raises(ValueError) as excinfo:
            RosTelemetryBase._validate_joint_limits({"elbow": (-1.9, _NAN)})
        assert str(excinfo.value) == finite_number_error(_NAN, "max", "joint_limits['elbow']")

    def test_finiteness_is_decided_before_the_ordering_comparison(self) -> None:
        """``(inf, -inf)`` trips both checks; the finite reason is the actionable one."""
        with pytest.raises(ValueError) as excinfo:
            RosTelemetryBase._validate_joint_limits({"j": (_INF, -_INF)})
        message = str(excinfo.value)
        assert "min must be a finite number" in message
        assert "> max" not in message

    def test_a_bound_past_the_float64_range_answers_rather_than_overflowing(self) -> None:
        """``float()`` raised ``OverflowError`` here, outside the documented contract."""
        huge = 10**400
        with pytest.raises(ValueError, match="min must be within the range of a 64-bit float"):
            RosTelemetryBase._validate_joint_limits({"j": (huge, 1.0)})

    def test_an_infinite_bound_is_refused_rather_than_read_as_an_open_side(self) -> None:
        """A half-infinite pair declares protection it does not provide.

        ``(-1.9, inf)`` and ``(-inf, 1.9)`` did admit in-range commands before
        this rule, so refusing them is a deliberate narrowing rather than a
        no-op: a ``{"<motor>.pos": (min, max)}`` clamp range is a bounded
        interval, an
        unbounded joint is expressed by omitting it (documented: "Joints without
        a declared bound are not constrained"), and the sibling declaration of
        this parameter refuses ``inf`` too - one parameter name must not carry
        two domains.
        """
        for bounds in ((-1.9, _INF), (-_INF, 1.9), (-_INF, _INF)):
            with pytest.raises(ValueError, match="must be a finite number"):
                RosTelemetryBase._validate_joint_limits({"j": bounds})

    def test_a_finite_range_still_admits_and_still_rejects(self) -> None:
        """The over-reach control: the feature the bounds exist for is untouched."""
        limits = RosTelemetryBase._validate_joint_limits({"shoulder_pan": (-1.9, 1.9)})
        assert limits == {"shoulder_pan": (-1.9, 1.9)}
        base = RosTelemetryBase()
        assert base._command_action(_Msg(["shoulder_pan"], [0.5]), joint_limits=limits) == {"shoulder_pan": 0.5}
        assert base._command_action(_Msg(["shoulder_pan"], [2.5]), joint_limits=limits) is None
        # A degenerate single-point range is still a range: it admits its own value.
        single = RosTelemetryBase._validate_joint_limits({"j": (1.9, 1.9)})
        assert base._command_action(_Msg(["j"], [1.9]), joint_limits=single) == {"j": 1.9}

    def test_both_hardware_bridges_inherit_the_validated_surface(self) -> None:
        """One validator covers both transports, so neither can drift."""
        from strands_robots.hardware_ros_bridge import HardwareRosBridge
        from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge

        for bridge in (HardwareRosBridge, HardwareRtpsBridge):
            assert issubclass(bridge, RosTelemetryBase)
            assert bridge._validate_joint_limits is RosTelemetryBase._validate_joint_limits
