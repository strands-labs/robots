"""The local teleop loop holds a leader frame to the same slew bound as the mesh.

``teleoperate(publish=True)`` drives a local follower and, from the same
``get_action()`` stream, every remote one. The mesh receive path bounds each
inbound frame's per-joint speed; the local path uses ``STRANDS_TELEOP_SLEW_ABS``
merge+apply loop applied its frames straight to ``send_action``. So one device
was judged by two different rules, and the follower physically next to the
operator was the unguarded one.

These tests pin that a frame no remote follower would accept is not applied to
a local one either, that a stream a physical leader arm can actually produce is
untouched, and that a refusal is visible rather than silently reported as a
clean run.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from strands_robots import teleop_mixin
from strands_robots.mesh import security
from tests.test_teleop import FakeHost, FakeTeleop

#: Loop rate every test drives, so the charged interval is floored at 1/50 s.
HZ = 50.0

#: One frame of a full-scale 12-bit encoder glitch, in the driver units a
#: leader arm actually streams. Over one 1/50 s tick this is 102400
#: units/second, 200x the local default bound - a speed no servo produces.
GLITCH = -2048.0


class SteppingLeader:
    """Emits a scripted sequence of values for one joint, then holds the last."""

    name, id, is_connected = "leader", None, False

    def __init__(self, values: list[float], joint: str = "joint1") -> None:
        self.values = values
        self.joint = joint
        self.calls = 0

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        i = min(self.calls, len(self.values) - 1)
        self.calls += 1
        return {self.joint: self.values[i]}


def _drive(host: FakeHost, leader: object, ticks: int, hz: float = HZ) -> dict:
    host.attach_teleop(leader, name="leader")
    return host.teleoperate(block=True, hz=hz, duration=ticks / hz)


class TestAnOverSpeedFrameIsNotApplied:
    def test_a_full_scale_jump_never_reaches_send_action(self) -> None:
        # A leader that reads one full-scale value - an encoder glitch, a USB
        # re-enumerate - then returns to rest, at a speed no servo produces.
        host = FakeHost()
        _drive(host, SteppingLeader([0.0, 0.0, GLITCH, 0.0, 0.0, 0.0]), ticks=12)

        applied = [a["joint1"] for a, _ in host.sent]
        assert applied, "premise: the loop must apply something"
        assert GLITCH not in applied, f"the glitched frame was applied: {applied}"

    def test_the_refusal_is_counted_and_reported_not_silent(self) -> None:
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 0.0, GLITCH, 0.0, 0.0, 0.0]), ticks=12)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] >= 1
        # A refusal is not an error: nothing failed.
        assert telemetry["errors"] == 0
        assert "refused" in result["content"][0]["text"]

    def test_a_refused_frame_does_not_become_the_next_frame_baseline(self) -> None:
        # After refusing the jump, the loop must keep measuring from the last
        # value it actually applied, so the stream resumes on its own.
        host = FakeHost()
        _drive(host, SteppingLeader([0.0, 0.0, GLITCH, 0.01, 0.02, 0.03]), ticks=14)

        applied = [a["joint1"] for a, _ in host.sent]
        assert 0.03 in applied, f"the stream never resumed: {applied}"


class TestTheBoundIsTheMeshBoundNotACopy:
    def test_the_loop_consults_the_mesh_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Single source of truth: the local path must call the mesh path's own
        # helper, so the two cannot drift to different rules for one device.
        seen: list[dict[str, float]] = []

        def spy(action, previous, now_mono, min_interval_s, max_slew=None):  # noqa: ANN001, ANN202
            seen.append(dict(action))
            return None

        monkeypatch.setattr(security, "input_frame_slew_violation", spy)
        host = FakeHost()
        _drive(host, FakeTeleop({"joint1": 0.1}), ticks=6)

        assert seen, "the local loop never consulted the shared slew helper"
        assert all("joint1" in frame for frame in seen)

    def test_the_operator_env_knob_widens_the_local_bound_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # STRANDS_TELEOP_SLEW_ABS is the documented way to widen the local
        # bound for a device whose driver units exceed the 500 units/s default.
        # (The mesh path uses its own STRANDS_MESH_INPUT_SLEW_ABS.)
        monkeypatch.setenv("STRANDS_TELEOP_SLEW_ABS", "200000")
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 0.0, GLITCH, 0.0, 0.0, 0.0]), ticks=12)

        assert result["content"][1]["json"]["slew_rejected"] == 0
        assert GLITCH in [a["joint1"] for a, _ in host.sent]


class TestAPhysicalLeaderIsUntouched:
    def test_a_stream_at_servo_speed_is_applied_in_full(self) -> None:
        # The bound is a speed above what a leader arm's own servos produce, so
        # a real stream must pass unchanged. 0.1 units per 1/50 s tick is
        # 5 units/s, under a fifth of the bound.
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), ticks=12)

        assert result["status"] == "success"
        assert result["content"][1]["json"]["slew_rejected"] == 0
        applied = [a["joint1"] for a, _ in host.sent]
        assert 0.5 in applied, f"a servo-speed stream was throttled: {applied}"

    def test_a_clean_session_text_names_no_refusal(self) -> None:
        host = FakeHost()
        result = _drive(host, FakeTeleop({"joint1": 0.1}), ticks=8)

        assert "refused" not in result["content"][0]["text"]

    def test_the_very_first_frame_is_always_applied(self) -> None:
        # There is no baseline to measure the first frame against, so it cannot
        # be refused however far from the follower's pose it reaches.
        host = FakeHost()
        _drive(host, SteppingLeader([GLITCH]), ticks=4)

        assert host.sent, "the first frame was refused with no baseline to judge it"
        assert host.sent[0][0]["joint1"] == GLITCH


class TestRefusalsAreVisibleInTheSessionStatus:
    def test_a_wholly_refused_session_does_not_report_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A device whose units the bound does not expect has every frame
        # refused. 0 frames / 0 errors used to derive "success": a silent
        # no-op, which is what this derivation exists to refuse.
        monkeypatch.setenv("STRANDS_TELEOP_SLEW_ABS", "0.0001")
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), ticks=12)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] >= 1
        assert result["status"] != "success", telemetry

    def test_a_partially_refused_session_is_degraded(self) -> None:
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 0.0, GLITCH, 0.0, 0.0, 0.0]), ticks=12)

        assert result["status"] == "degraded"
        assert result["content"][1]["json"]["frames"] > 0

    def test_the_live_status_surface_reports_refusals(self) -> None:
        host = FakeHost()
        _drive(host, SteppingLeader([0.0, 0.0, GLITCH, 0.0, 0.0, 0.0]), ticks=12)

        live = host.get_teleoperate_status()
        assert live["content"][1]["json"]["slew_rejected"] >= 1
        assert "slew_rejected" in live["content"][0]["text"]

    def test_the_baseline_does_not_carry_across_sessions(self) -> None:
        # A new session starts with no baseline, so its first frame is applied
        # rather than measured against wherever the previous session ended.
        host = FakeHost()
        _drive(host, SteppingLeader([2.0]), ticks=4)
        host.stop_teleoperate()
        host.detach_teleop()

        host.sent.clear()
        result = _drive(host, SteppingLeader([-2.0]), ticks=4)

        assert host.sent[0][0]["joint1"] == -2.0
        assert result["content"][1]["json"]["slew_rejected"] == 0


class TestTheMixinStaysLight:
    """The shared bound must not drag a layer this module may not depend on.

    ``strands_robots.teleop_mixin`` must not depend on
    ``strands_robots.simulation`` - that separation is why the shared numeric
    domains live in :mod:`strands_robots.utils` (see
    :func:`strands_robots.utils.positive_finite_number_error`). Importing
    ``strands_robots.mesh.security`` executes the ``mesh`` package, which does
    reach ``strands_robots.simulation``, so the slew helper has to be imported
    inside the loop. Hoisting it to module scope would still pass every test
    above while quietly inverting the layering, so it is pinned here.
    """

    @staticmethod
    def _module_scope_imports() -> set[str]:
        source = inspect.getsource(teleop_mixin)
        tree = ast.parse(source)
        names: set[str] = set()
        for node in tree.body:  # module scope only, not nested in a def
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names.add(node.module)
        return names

    def test_the_mesh_package_is_not_imported_at_module_scope(self) -> None:
        offenders = sorted(
            name
            for name in self._module_scope_imports()
            if name.startswith(("strands_robots.mesh", "strands_robots.simulation"))
        )
        assert not offenders, (
            f"{offenders} imported at module scope: importing the mesh package reaches "
            f"strands_robots.simulation, which this module must not depend on. Import the "
            f"slew helper inside _teleop_loop instead."
        )

    def test_the_slew_helper_is_imported_inside_the_loop(self) -> None:
        # The converse of the test above: proves the bound is reached lazily
        # rather than not reached at all.
        source = inspect.getsource(teleop_mixin.TeleopMixin._teleop_loop)
        assert "from strands_robots.mesh.security import" in source
        assert "input_frame_slew_violation" in source


class TestDefaultBoundAccommodatesDriverUnits:
    """The default local slew bound (500 units/s) must accommodate degree-valued
    and range-0-100 devices at their shipped defaults without env-var tuning.

    These are the streams that `robot.attach_teleop("so101_leader", port=...).
    teleoperate()` produces: joints in degrees, gripper in 0-100 range.
    """

    def test_a_90_degree_sweep_over_1s_is_not_refused(self) -> None:
        # A calm 90-degree arm sweep at 50 Hz: each tick moves 1.8 degrees,
        # producing 90 deg/s peak speed. The 500 units/s default must accept it.
        positions = [i * 1.8 for i in range(50)]  # 0.0, 1.8, ... 88.2
        host = FakeHost()
        result = _drive(host, SteppingLeader(positions), ticks=50)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] == 0, (
            f"a 90 deg/s degree-valued stream was refused at the default bound: {telemetry}"
        )

    def test_a_half_second_gripper_close_is_not_refused(self) -> None:
        # Gripper in range-0-100: close from 0 to 100 in 0.5 s at 50 Hz is
        # 25 ticks of 4 units each = 200 units/s peak. Must be accepted.
        positions = [i * 4.0 for i in range(25)]  # 0, 4, 8, ... 96
        host = FakeHost()
        result = _drive(host, SteppingLeader(positions), ticks=25)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] == 0, (
            f"a 200 units/s gripper close was refused at the default bound: {telemetry}"
        )

    def test_sts3215_no_load_max_in_degrees_is_not_refused(self) -> None:
        # STS3215 no-load max is 6.5 rad/s = ~372 deg/s. At 50 Hz that is
        # 7.44 deg/tick. Must be accepted at the default bound.
        positions = [i * 7.44 for i in range(20)]
        host = FakeHost()
        result = _drive(host, SteppingLeader(positions), ticks=20)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] == 0, (
            f"a 372 deg/s stream (STS3215 max) was refused at the default bound: {telemetry}"
        )

    def test_a_2000_units_per_second_glitch_is_still_refused(self) -> None:
        # An encoder glitch that jumps 40 units in one tick at 50 Hz =
        # 2000 units/s. This MUST still be caught even at the wider default.
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 0.0, 40.0, 0.0, 0.0, 0.0]), ticks=12)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] >= 1, (
            f"a 2000 units/s glitch was NOT refused at the default bound: {telemetry}"
        )


class QuietThenGlitchLeader:
    """A device that stamps one joint, goes quiet, then returns full-scale.

    The shape of a USB re-enumerate or a disconnect in a multi-device session:
    ``get_action()`` returns ``{}`` for a while - the device commands nothing,
    so nothing of its own is applied - and its first read on return is a
    full-scale garbage value.
    """

    name, id, is_connected = "quiet", None, False

    def __init__(self, first: float, quiet_ticks: int, comeback: float, joint: str = "joint1") -> None:
        self.first, self.quiet_ticks, self.comeback, self.joint = first, quiet_ticks, comeback, joint
        self.calls = 0

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        self.calls += 1
        if self.calls == 1:
            return {self.joint: self.first}
        if self.calls <= 1 + self.quiet_ticks:
            return {}  # quiet: commands nothing
        return {self.joint: self.comeback}


class SteadyLeader:
    """Holds one joint at a constant value, so frames keep being applied."""

    name, id, is_connected = "steady", None, False

    def __init__(self, value: float = 0.0, joint: str = "joint2") -> None:
        self.value, self.joint = value, joint

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        return {self.joint: self.value}


class TestAQuietDeviceKeepsItsBaselineWhileOthersMove:
    """A joint that stops being commanded must not lose the entry that guards it.

    ``merge_slew_baseline`` prunes an entry once it "can no longer refuse
    anything" - a horizon computed from the bound it is *passed*. Parameterised
    with the mesh defaults while the check runs at the local bound, the horizon
    is 0.5 s for a joint resting near zero, but under the local bound that same
    entry can still refuse a full-scale glitch for seconds. Entries were
    therefore dropped while they could still refuse frames, and the prune only
    fires when *some other* device keeps a session applying frames - which is
    why every single-device test above passes either way.
    """

    @staticmethod
    def _drive_two(host: FakeHost, quiet: object, steady: object, ticks: int) -> dict:
        host.attach_teleop(quiet, name="quiet")
        host.attach_teleop(steady, name="steady")
        return host.teleoperate(block=True, hz=HZ, duration=ticks / HZ)

    def test_a_glitch_from_a_returning_device_is_still_refused(self) -> None:
        # joint1 rests at 0.0, then its device goes quiet for 30 ticks (0.6 s,
        # past the 0.5 s mesh-default horizon for a joint at rest) while joint2
        # keeps the loop applying frames. joint1's comeback read is full-scale.
        host = FakeHost()
        quiet = QuietThenGlitchLeader(first=0.0, quiet_ticks=30, comeback=GLITCH)
        result = self._drive_two(host, quiet, SteadyLeader(), ticks=40)

        # Premise: the loop kept applying frames through the quiet window, so
        # the prune was actually exercised. Without this the test proves nothing.
        assert len(host.sent) > 30, f"the prune was never exercised: {len(host.sent)} frames"

        applied_j1 = [a["joint1"] for a, _ in host.sent if "joint1" in a]
        assert GLITCH not in applied_j1, (
            f"the returning device's full-scale frame was applied: {applied_j1}. Its baseline "
            f"entry was pruned at the mesh horizon while the check runs at the local bound."
        )
        assert result["content"][1]["json"]["slew_rejected"] >= 1, (
            f"the glitch passed with no refusal counted: {result['content'][1]['json']}"
        )

    def test_the_session_does_not_report_success_for_a_refused_glitch(self) -> None:
        # The failure this pins was silent: the frame applied, nothing counted,
        # status still success. Status must reflect the refusal.
        host = FakeHost()
        quiet = QuietThenGlitchLeader(first=0.0, quiet_ticks=30, comeback=GLITCH)
        result = self._drive_two(host, quiet, SteadyLeader(), ticks=40)

        assert "refused" in result["content"][0]["text"]


class TestBothCallSitesCarryTheLocalBound:
    """Neither mesh helper may be called on its own defaults from this loop.

    This is the class of bug, not one instance of it: the loop enforces a local
    bound, and every mesh helper it calls has to be told so, because each one
    silently falls back to the mesh path's radian-scoped defaults. The checker
    inheriting them refuses ordinary degree-valued teleop; the pruner inheriting
    them discards entries that can still refuse. Both are invisible in review -
    the call reads fine - so the rule is pinned over the AST rather than left to
    a reviewer noticing a missing keyword.
    """

    #: Helper to the parameters that describe the bound it must be judged by.
    _REQUIRED: dict[str, set[str]] = {
        "input_frame_slew_violation": {"max_slew"},
        "merge_slew_baseline": {"max_slew", "value_abs"},
    }

    def test_every_mesh_slew_helper_call_passes_the_local_parameters(self) -> None:
        import textwrap

        source = textwrap.dedent(inspect.getsource(teleop_mixin.TeleopMixin._teleop_loop))
        tree = ast.parse(source)

        found: dict[str, int] = {}
        offenders: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            required = self._REQUIRED.get(node.func.id)
            if required is None:
                continue
            found[node.func.id] = found.get(node.func.id, 0) + 1
            passed = {kw.arg for kw in node.keywords if kw.arg is not None}
            missing = sorted(required - passed)
            if missing:
                offenders.append(f"{node.func.id} (line {node.lineno}) omits {missing}")

        # Premise: both helpers are actually called, so the pin cannot pass by
        # matching nothing if the loop is refactored.
        assert set(found) == set(self._REQUIRED), (
            f"expected calls to {sorted(self._REQUIRED)} in _teleop_loop, found {sorted(found)}"
        )
        assert not offenders, (
            f"mesh slew helper(s) called on mesh defaults from the local loop: {offenders}. "
            f"Each falls back to STRANDS_MESH_INPUT_* - a bound this path does not enforce - "
            f"so pass the local bound explicitly at every call site."
        )

    def test_the_prune_envelope_is_unbounded_because_the_path_has_no_clamp(self) -> None:
        # The envelope is what makes the prune safe: it is the furthest a
        # permissible command may reach. The local path runs no magnitude clamp,
        # so a finite envelope would prune entries that can still refuse.
        source = inspect.getsource(teleop_mixin.TeleopMixin._teleop_loop)
        assert "math.inf" in source, (
            "the local prune envelope must be unbounded: input_frame_slew_violation applies no "
            "magnitude clamp, so no finite displacement makes a baseline entry unable to refuse."
        )


class TestNarrowingTheBoundDoesNotShortenTheBaseline:
    """Narrowing the local bound widens the window an entry can still refuse in.

    ``merge_slew_baseline`` keeps an entry for ``(value_abs + abs(value)) /
    max_slew`` seconds. A *narrower* bound makes that horizon longer, so an
    operator who tightens ``STRANDS_TELEOP_SLEW_ABS`` needs the prune to track
    the tightening too - otherwise the entries it relies on are discarded
    earlier, relative to the bound in force, than at the default.
    """

    #: How long the mesh defaults keep a resting joint's entry: ``value_abs /
    #: max_slew`` seconds. A gap longer than this drops the entry when the prune
    #: runs on those defaults.
    MESH_HORIZON_AT_REST_S = security._input_value_abs() / security._input_slew_abs()

    def test_a_returning_device_is_refused_under_a_narrowed_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_TELEOP_SLEW_ABS", "5.0")
        quiet_ticks = 40
        assert quiet_ticks / HZ > self.MESH_HORIZON_AT_REST_S, (
            "premise: the gap must outlast the mesh prune horizon, or the entry survives "
            "for reasons unrelated to which parameters the prune was given"
        )

        host = FakeHost()
        host.attach_teleop(QuietThenGlitchLeader(0.0, quiet_ticks, GLITCH), name="quiet")
        host.attach_teleop(SteadyLeader(), name="steady")
        ticks = 2 + quiet_ticks + 4
        result = host.teleoperate(block=True, hz=HZ, duration=ticks / HZ)

        applied = [a["joint1"] for a, _ in host.sent if "joint1" in a]
        assert applied, "premise: the quiet device must have applied its resting frames"
        assert GLITCH not in applied, (
            f"a full-scale jump was applied under a narrowed bound after a {quiet_ticks / HZ:.2f}s gap: {applied}"
        )
        assert result["content"][1]["json"]["slew_rejected"] >= 1

    def test_the_baseline_only_holds_joints_that_reached_the_robot(self) -> None:
        # The local prune is a no-op, so what bounds the baseline is that an
        # entry is only ever stamped for a key that was applied - the claim the
        # unbounded envelope rests on.
        host = FakeHost()
        host.attach_teleop(QuietThenGlitchLeader(0.0, 10, GLITCH), name="quiet")
        host.attach_teleop(SteadyLeader(), name="steady")
        host.teleoperate(block=True, hz=HZ, duration=16 / HZ)

        applied_keys = {k for a, _ in host.sent for k in a}
        assert applied_keys, "premise: the loop must have applied something"
        assert set(host._teleop_slew_baseline) <= applied_keys, (
            f"the baseline holds keys that never reached the robot: "
            f"{sorted(set(host._teleop_slew_baseline) - applied_keys)}"
        )


class TestTheMeshPathKeepsItsOwnPruneHorizon:
    """Parameterising the local call site changes nothing for the mesh path."""

    def test_called_without_overrides_the_prune_still_uses_the_mesh_horizon(self) -> None:
        # This is how the mesh receive path calls it: no overrides, so the
        # horizon stays the one its own magnitude clamp justifies.
        horizon = security._input_value_abs() / security._input_slew_abs()
        at_rest = {"joint1": (0.0, 1000.0)}

        assert "joint1" in security.merge_slew_baseline(at_rest, {}, 1000.0 + horizon * 0.9)
        assert "joint1" not in security.merge_slew_baseline(at_rest, {}, 1000.0 + horizon * 1.1)
