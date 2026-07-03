"""The shared bridge base fails a mis-secured inbound command surface at build time.

An inbound ``joint_command`` subscription lets any participant on the DDS domain
drive a physical arm, so :class:`strands_robots.ros_telemetry.RosTelemetryBase`
validates the drivable surface at construction rather than silently degrading:

* ``joint_limits`` must be a well-formed ``{motor: (min, max)}`` mapping, or the
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
        {"a": ("low", "high")},  # non-numeric bound
        {"a": 5.0},  # not a pair at all
    ],
)
def test_malformed_bound_pairs_are_rejected_naming_the_joint(bad: dict[str, object]) -> None:
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
