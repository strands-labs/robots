"""``STRANDS_MESH_STREAM_HZ`` must not be divided straight from the environment.

Two call sites throttle ``publish_step`` against the wall clock -- the hardware
control loop and the simulation ``run_policy`` hook -- and both computed their
period as ``1.0 / float(os.environ.get("STRANDS_MESH_STREAM_HZ", "10") or 10)``.
That expression raises ``ZeroDivisionError`` on ``0`` and ``ValueError`` on any
non-numeric value. One of the two runs inside ``HardwareRobot.__init__``, so the
consequence is not "telemetry degraded" but "every ``Robot(..., mode="real")``
construction raises before the serial port is even opened".

``0`` is a realistic input, not an adversarial one. The sibling knob
``STRANDS_MESH_CAMERA_HZ`` advertises non-positive as off -- ``Mesh.start``
tells operators to *set* it to enable frames -- so an operator disabling step
telemetry the same way bricked hardware bring-up.

Both sites now route through :func:`stream_min_period_from_env`, which reuses
the package's shared ``hz_from_env`` domain check and expresses "off" as
``math.inf``: no elapsed wall-clock time reaches an infinite period, so the
throttle never fires and no caller needs a second flag.

:class:`TestNeitherCallSiteDividesTheRawEnvValue` is the structural half. The
behavioural tests would pass again if someone reintroduced the division at a
third site, and the bug was a duplicated expression in the first place, so the
source itself is asserted on.
"""

from __future__ import annotations

import ast
import math
import pathlib

import pytest

from strands_robots.mesh.session import STREAM_HZ, stream_min_period_from_env

#: Modules that hold a ``publish_step`` throttle period.
THROTTLE_SITES = [
    "strands_robots/hardware_robot.py",
    "strands_robots/simulation/mujoco/simulation.py",
]


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


class TestAnUnsetVariableKeepsTheDocumentedDefault:
    """The common case is unchanged by the guard."""

    def test_unset_resolves_to_the_stream_hz_period(self, monkeypatch) -> None:
        monkeypatch.delenv("STRANDS_MESH_STREAM_HZ", raising=False)

        assert stream_min_period_from_env() == pytest.approx(1.0 / STREAM_HZ)

    def test_blank_is_treated_as_unset(self, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", "   ")

        assert stream_min_period_from_env() == pytest.approx(1.0 / STREAM_HZ)

    @pytest.mark.parametrize(("raw", "period"), [("5", 0.2), ("20", 0.05), ("0.5", 2.0)])
    def test_a_positive_rate_becomes_its_period(self, monkeypatch, raw, period) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", raw)

        assert stream_min_period_from_env() == pytest.approx(period)


class TestANonPositiveRateTurnsPublishingOff:
    """``0`` is the operator opt-out, spelled as it is for the camera loop."""

    @pytest.mark.parametrize("raw", ["0", "0.0", "-1", "-0.5"])
    def test_it_resolves_to_an_infinite_period(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", raw)

        assert stream_min_period_from_env() == math.inf

    def test_an_infinite_period_never_lets_the_throttle_fire(self, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", "0")
        period = stream_min_period_from_env()

        # The shape both call sites use: ``now - last >= period``. A day of
        # elapsed time must not clear an infinite period.
        assert not (86_400.0 - 0.0 >= period)


class TestAnUnusableValueIsReportedAndDisablesPublishing:
    """Follows ``_resolve_camera_hz``: warn, then leave the loop off."""

    @pytest.mark.parametrize("raw", ["fast", "10Hz", "", "nan", "inf", "1e999"])
    def test_it_does_not_raise(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", raw)

        # No exception: the point of the change is that a misconfigured knob
        # cannot be the thing that fails a constructor.
        stream_min_period_from_env()

    @pytest.mark.parametrize("raw", ["fast", "nan", "inf", "1e999"])
    def test_it_resolves_to_an_infinite_period(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", raw)

        assert stream_min_period_from_env() == math.inf

    def test_the_warning_names_the_variable_and_the_value(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", "fast")

        with caplog.at_level("WARNING", logger="strands_robots.mesh.session"):
            stream_min_period_from_env()

        # "no step telemetry" is undiagnosable without both halves in the log.
        assert "STRANDS_MESH_STREAM_HZ" in caplog.text
        assert "fast" in caplog.text

    def test_a_non_positive_value_is_not_warned_about(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", "0")

        with caplog.at_level("WARNING", logger="strands_robots.mesh.session"):
            stream_min_period_from_env()

        # 0 is a request, not a mistake; warning about it trains operators to
        # ignore the warning that does mean something.
        assert caplog.text == ""


class TestNeitherCallSiteDividesTheRawEnvValue:
    """The duplicated expression cannot come back unnoticed."""

    @pytest.mark.parametrize("relpath", THROTTLE_SITES)
    def test_the_module_does_not_read_the_knob_directly(self, relpath) -> None:
        source = (_repo_root() / relpath).read_text(encoding="utf-8")
        tree = ast.parse(source)

        # A string literal naming the knob outside a comment means the module
        # resolved it itself instead of calling the shared resolver.
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == "STRANDS_MESH_STREAM_HZ"
        ]

        assert not literals, (
            f"{relpath} names STRANDS_MESH_STREAM_HZ itself; the period belongs to "
            "strands_robots.mesh.session.stream_min_period_from_env, which applies "
            "the shared hz_from_env domain check"
        )

    @pytest.mark.parametrize("relpath", THROTTLE_SITES)
    def test_the_module_calls_the_shared_resolver(self, relpath) -> None:
        source = (_repo_root() / relpath).read_text(encoding="utf-8")

        assert "stream_min_period_from_env()" in source
