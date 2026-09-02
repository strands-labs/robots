"""The lifecycle over a hardware-shaped fake link: arm, stream, cache, release.

Four properties that only appear once the driver is driving something:

* **Arming is not optional.** Firmware 2023.02 and later refuse to spin the
  motors until ``Platform.send_arming_request(True)`` succeeds. A driver that
  connected without arming would accept every setpoint and produce no motion -
  the quietest possible failure - so an unarmed driver refuses to command.
* **A setpoint is a stream.** The firmware supervisor cuts thrust when the
  setpoint stream goes quiet, so one ``send_action`` has to keep sending. The pin
  is that the *same* setpoint arrives repeatedly without a second call.
* **Telemetry is cached, not load-bearing.** A log block that cannot start
  leaves the aircraft flyable, so it is reported and not fatal; a block that does
  start populates the three attributes the mesh reads by ``getattr``.
* **Release is complete.** After ``cleanup`` the link is closed and the driver
  reports itself disconnected rather than looking live with nothing behind it.
* **A link is only open once the aircraft says so.** ``cflib``'s ``open_link``
  is asynchronous and never raises, so its return says nothing; the outcome
  arrives on the link thread. A driver that read the return would arm and fly a
  vehicle that is not there, because ``send_packet`` discards every packet in
  silence while there is no link.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from strands_robots.drivers import crazyflie as module

#: The package every module :func:`~strands_robots.drivers.crazyflie._resolve_cflib`
#: imports lives under, and the distribution the ``[crazyflie]`` extra supplies.
#: Blocking this one entry blocks all of them whatever submodules the resolver
#: reaches for, because importing a submodule imports its parent package first.
_CFLIB_PACKAGE = "cflib"


def _block_cflib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``cflib`` unimportable for the current test.

    A ``None`` entry in :data:`sys.modules` turns the resolver's
    :func:`importlib.import_module` call into an :exc:`ImportError`, which is the
    only branch that produces a reason. Forcing it is what lets the reason be
    graded on a machine that *has* the extra installed. Only that one entry is
    touched, and ``monkeypatch`` restores it after the test.

    Args:
        monkeypatch: pytest's patcher, which restores the entry after the test.
    """
    monkeypatch.setitem(sys.modules, _CFLIB_PACKAGE, None)


class TestConnectingArmsTheAircraft:
    """Nothing flies until the arming request succeeds."""

    def test_a_successful_connect_arms_and_reports_no_reason(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        driver, fake, reason = connected()
        assert reason is None
        assert driver.is_connected
        assert recorder.args_of("platform.send_arming_request") == (True,)
        assert fake.uri == module.DEFAULT_URI, "the default URI is the one a stock Crazyflie ships with"

    def test_the_port_keyword_is_the_link_uri(self, connected) -> None:  # type: ignore[no-untyped-def]
        _, fake, reason = connected(port="usb://0")
        assert reason is None
        assert fake.uri == "usb://0", "port= stays polymorphic - here the USB cable rather than a radio"

    def test_an_arming_failure_is_reported_and_blocks_every_command(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """The failure mode this guards is an aircraft that accepts commands silently."""
        driver, _, reason = connected(arming=RuntimeError("no ack"))
        assert reason is not None
        assert "arming" in reason and "no ack" in reason

        envelope = driver.send_action({"vx": 0.2, "z": 0.5})
        assert envelope["status"] == "error"
        assert "not armed" in envelope["content"][0]["text"]
        assert recorder.count("commander.send_hover_setpoint") == 0

        assert driver.takeoff()["status"] == "error"
        assert recorder.count("high_level.takeoff") == 0

    def test_arming_precedes_the_telemetry_block(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """A vehicle that cannot be armed should not have a log block started for it."""
        connected()
        names = recorder.names()
        assert names.index("platform.send_arming_request") < names.index("log.start")


class TestConnectingWaitsForTheLinkToComeUp:
    """``open_link`` returning is not the aircraft answering.

    ``Crazyflie.open_link`` wraps its body in ``except Exception`` and routes
    every failure to the ``connection_failed`` *callback*, so with no Crazyradio
    plugged in it returns normally and leaves ``cf.link`` at ``None``. Nothing
    downstream complains either: ``Crazyflie.send_packet`` is a silent no-op
    while ``link`` is ``None``, so the arming request, every setpoint and every
    ``takeoff`` would be discarded while this driver answered ``success``. These
    pin that the driver waits for the outcome and reports it.
    """

    def test_a_refused_link_is_reported_rather_than_reported_as_connected(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """The absent-dongle case: the SDK reports it by callback, not by raising."""
        driver, _, reason = connected(outcome="failed", failure="Cannot find a Crazyradio Dongle")

        assert reason is not None, (
            "open_link returned normally but the link never came up; reporting None here would "
            "leave the caller flying an aircraft that is not there"
        )
        assert "Cannot find a Crazyradio Dongle" in reason, f"the SDK's own reason must survive: {reason}"
        assert module.DEFAULT_URI in reason, f"a reason that does not name the URI is not actionable: {reason}"
        assert driver.is_connected is False

    def test_the_reason_is_one_actionable_line_not_a_pasted_traceback(self, connected) -> None:  # type: ignore[no-untyped-def]
        """``cflib`` appends ``traceback.format_exc()`` to this message.

        The first line is the whole actionable content; the rest is an internal
        stack that would be pasted into an agent-facing error envelope verbatim.
        """
        sdk_message = "Couldn't load link driver: Cannot find a Crazyradio Dongle\n\nTraceback (most recent call last):\n  File 'radiodriver.py', line 224\nException: Cannot find a Crazyradio Dongle\n"
        _, _, reason = connected(outcome="failed", failure=sdk_message)

        assert reason is not None
        assert "Cannot find a Crazyradio Dongle" in reason, f"the actionable half was dropped: {reason}"
        assert "Traceback" not in reason, f"an internal stack reached the refusal: {reason}"
        assert reason.count("\n") == 0, f"a refusal must be one line: {reason!r}"

    def test_a_refused_link_is_never_armed(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """Arming a vehicle that is not there is the packet that races the setup."""
        connected(outcome="failed")
        assert recorder.count("platform.send_arming_request") == 0, (
            f"the arming request was sent over a link that never opened: {recorder.names()}"
        )
        assert recorder.count("log.add_config") == 0, "a telemetry block cannot be added before the TOC is down"

    def test_a_refused_link_refuses_every_flight_command(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """The whole point: no envelope may say success over a dropped packet."""
        driver, _, _ = connected(outcome="failed")

        for envelope in (driver.send_action({"vx": 0.2, "z": 0.5}), driver.takeoff(), driver.land()):
            assert envelope["status"] == "error", f"a command was accepted with no link: {envelope}"
        assert recorder.count("commander.send_hover_setpoint") == 0
        assert recorder.count("high_level.takeoff") == 0

    def test_a_refused_link_is_released_so_a_retry_can_have_the_dongle(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        connected(outcome="failed")
        assert recorder.count("close_link") == 1, (
            f"the failed link was left open, so the dongle stays claimed: {recorder.names()}"
        )

    def test_an_aircraft_that_never_answers_is_reported_after_the_bounded_wait(  # type: ignore[no-untyped-def]
        self, connected, recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dongle that enumerated and then went quiet. Only a timeout sees it.

        ``cflib``'s own ``SyncCrazyflie.open_link`` waits unbounded here, which
        for an agent is indistinguishable from a hang, so the driver bounds it.
        """
        monkeypatch.setattr(module, "CONNECT_TIMEOUT_S", 0.05)
        driver, _, reason = connected(outcome="silent")

        assert reason is not None, "an aircraft that never answered was reported as connected"
        assert "did not answer" in reason and module.DEFAULT_URI in reason, reason
        assert driver.is_connected is False
        assert recorder.count("platform.send_arming_request") == 0
        assert recorder.count("close_link") == 1

    def test_the_link_is_up_before_anything_is_sent_over_it(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """Ordering, on the happy path: ``connected`` precedes arming and the log block.

        The real ``log.add_config`` looks each variable up in the downloaded TOC
        and raises ``KeyError`` until ``connected`` has fired, so this ordering is
        also what makes telemetry work at all rather than land in the driver's
        degraded path on every real connection.

        The link is held back so the ordering is decided by the driver rather
        than by which thread happens to win: a driver that does not wait reaches
        the wire during the delay, a driver that waits cannot.
        """
        _, _, reason = connected(settle_delay=0.05)
        assert reason is None

        names = recorder.names()
        assert names.index("connected") < names.index("platform.send_arming_request"), (
            f"the arming request preceded the link coming up: {names}"
        )
        assert names.index("connected") < names.index("log.add_config"), (
            f"the telemetry block was added before the TOC was down: {names}"
        )


class TestTheSetpointStreamStaysAlive:
    """One ``send_action`` produces a stream, not a twitch."""

    def test_the_latched_setpoint_is_re_sent_without_a_second_call(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        driver, _, _ = connected(setpoint_hz=100)
        envelope = driver.send_action({"vx": 0.2, "z": 0.5})
        assert envelope["status"] == "success"

        reached = recorder.wait_for("commander.send_hover_setpoint", 3, timeout=5.0)
        driver.cleanup()
        assert reached, (
            "the setpoint was sent once and never repeated; the firmware supervisor cuts thrust "
            f"when the stream goes quiet. Calls: {recorder.names()}"
        )

    def test_every_repeat_carries_the_same_wire_arguments(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        driver, _, _ = connected(setpoint_hz=100)
        driver.send_action({"vx": 0.2, "z": 0.5})
        recorder.wait_for("commander.send_hover_setpoint", 3, timeout=5.0)
        driver.cleanup()

        sent = [args for name, args in recorder.calls if name == "commander.send_hover_setpoint"]
        assert len(set(sent)) == 1, f"the repeater must hold the setpoint, not drift: {sent}"
        assert sent[0] == (0.2, 0.0, 0.0, 0.5)

    def test_a_new_setpoint_replaces_the_latched_one(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        driver, _, _ = connected(setpoint_hz=100)
        driver.send_action({"vx": 0.2, "z": 0.5})
        driver.send_action({"vx": -0.3, "z": 0.8})
        recorder.wait_for("commander.send_hover_setpoint", 4, timeout=5.0)
        driver.cleanup()

        sent = [args for name, args in recorder.calls if name == "commander.send_hover_setpoint"]
        assert sent[-1] == (-0.3, 0.0, 0.0, 0.8), f"the repeater kept a stale setpoint: {sent}"

    def test_the_task_status_reports_the_setpoint_in_flight(self, connected) -> None:  # type: ignore[no-untyped-def]
        driver, _, _ = connected(setpoint_hz=100)
        assert driver.get_task_status()["content"][0]["json"]["streaming"] is False

        driver.send_action({"vx": 0.2, "z": 0.5})
        payload = driver.get_task_status()["content"][0]["json"]
        driver.cleanup()
        assert payload["streaming"] is True
        assert payload["commanded"] == module.HOVER_SETPOINT
        assert payload["args"] == [0.2, 0.0, 0.0, 0.5]

    def test_set_twist_is_the_same_command_by_another_name(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        driver, _, _ = connected(setpoint_hz=100)
        assert driver.set_twist(vx=0.2, wz=1.0, z=0.5)["status"] == "success"
        driver.cleanup()
        assert recorder.args_of("commander.send_hover_setpoint") == pytest.approx((0.2, 0.0, 57.29577951308232, 0.5))


class TestTelemetryIsCachedForTheMesh:
    """The three attributes the mesh reads with ``getattr(robot, name, None)``."""

    def test_the_log_block_subscribes_to_the_declared_variables(self, connected) -> None:  # type: ignore[no-untyped-def]
        _, fake, _ = connected()
        assert fake.log.block is not None
        assert tuple(fake.log.block.variables) == module.LOG_VARIABLES

    def test_a_delivered_frame_populates_pose_imu_and_battery(self, connected) -> None:  # type: ignore[no-untyped-def]
        driver, fake, _ = connected()
        assert driver._pose is None, "nothing is cached before a frame arrives"

        fake.log.block.deliver(
            {
                "stateEstimate.x": 0.1,
                "stateEstimate.y": -0.2,
                "stateEstimate.z": 0.55,
                "stabilizer.roll": 1.5,
                "stabilizer.pitch": -0.5,
                "stabilizer.yaw": 90.0,
                "pm.vbat": 3.85,
                "pm.state": 0,
            }
        )

        assert driver._pose == {"x": 0.1, "y": -0.2, "z": 0.55, "roll": 1.5, "pitch": -0.5, "yaw": 90.0}
        assert driver._imu == {"roll": 1.5, "pitch": -0.5, "yaw": 90.0}
        assert driver._battery == {"volts": 3.85, "state": "battery", "low_volts": module.LOW_BATTERY_VOLTS}

    def test_a_quadcopter_publishes_no_joints(self, connected) -> None:  # type: ignore[no-untyped-def]
        """Four propellers and no joint; an empty dict would be a reading."""
        driver, _, _ = connected()
        assert not hasattr(driver, "_joints")

    def test_a_telemetry_failure_leaves_the_aircraft_flyable(self, monkeypatch, recorder, connected) -> None:  # type: ignore[no-untyped-def]
        """A log variable this firmware lacks must not ground a usable vehicle."""

        def exploding_log_config(**kwargs: object) -> object:
            raise KeyError("stateEstimate.x")

        driver, fake, _ = connected()
        driver.cleanup()

        monkeypatch.setattr(
            module,
            "_resolve_cflib",
            lambda: type(
                "_P",
                (),
                {
                    "crtp": type("_C", (), {"init_drivers": staticmethod(lambda: None)}),
                    "Crazyflie": lambda **_: fake,
                    "LogConfig": exploding_log_config,
                },
            ),
        )
        second = module.CrazyflieDriver(setpoint_hz=100)
        assert second.connect_eagerly() is None, "telemetry is not load-bearing for flight"
        assert second.send_action({"z": 0.5})["status"] == "success"
        second.cleanup()


class TestReleasingTheDriver:
    """After cleanup nothing is held and nothing claims to be live."""

    def test_cleanup_stops_the_log_block_and_closes_the_link(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        driver, _, _ = connected()
        driver.cleanup()

        assert recorder.count("log.stop") == 1
        assert recorder.count("close_link") == 1
        assert driver.is_connected is False

    def test_cleanup_is_safe_on_a_driver_that_never_connected(self) -> None:
        """Cleanup is what runs after a failed connect, so it must tolerate one."""
        module.CrazyflieDriver().cleanup()

    def test_stop_on_a_disconnected_driver_does_not_raise(self) -> None:
        asyncio.run(module.CrazyflieDriver().stop())


class TestTheDegradedSurfaceWithoutCflib:
    """No radio library, no radio: reported, never raised."""

    def test_connect_reports_the_missing_module_and_the_remedy(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr(module, "_resolve_cflib", lambda: "cannot import cflib (no module); pip install x")
        driver = module.CrazyflieDriver()
        reason = driver.connect_eagerly()

        assert reason is not None and "cflib" in reason
        assert driver.is_connected is False

    def test_the_real_resolver_names_the_extra_that_supplies_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The remedy is graded with the extra installed, by forcing the absence.

        Skipping this wherever ``cflib`` imports would retire the pin in exactly the
        environments that ship the driver - the ``[all]`` feature set installs
        ``strands-robots[crazyflie]`` - so the one reason a user ever reads would go
        ungraded there. Blocking the import grades it everywhere instead.
        """
        _block_cflib(monkeypatch)
        reason = module._resolve_cflib()

        assert isinstance(reason, str), "a blocked cflib must report a reason, not resolve pieces"
        assert _CFLIB_PACKAGE in reason, "the reason must name the module that failed"
        assert "strands-robots[crazyflie]" in reason, "the reason must name the extra that fixes it"

    def test_the_resolver_answers_with_the_pieces_when_the_extra_is_present(self) -> None:
        """The control for the branch above, graded wherever the extra is installed."""
        resolved = module._resolve_cflib()
        if isinstance(resolved, str):
            pytest.skip(f"cflib is absent here, so the success branch is unreachable: {resolved}")
        for attribute in ("crtp", "Crazyflie", "LogConfig"):
            assert hasattr(resolved, attribute), f"the driver reads .{attribute} off the resolved pieces"

    def test_every_command_refuses_while_disconnected(self) -> None:
        driver = module.CrazyflieDriver()
        for envelope in (
            driver.send_action({"vx": 0.2}),
            driver.set_twist(vx=0.2),
            driver.takeoff(),
            driver.land(),
            driver.emergency_stop(),
        ):
            assert envelope["status"] == "error"
            assert "not connected" in envelope["content"][0]["text"]

    def test_status_is_answerable_with_no_link_at_all(self) -> None:
        """The mesh publishes presence for a peer that has not connected."""
        payload = asyncio.run(module.CrazyflieDriver().get_status())["content"][0]["json"]
        assert payload["connected"] is False
        assert payload["armed"] is False
        assert payload["envelope"] == module.twist_envelope()
        assert payload["supported_robots"] == list(module.SUPPORTED_ROBOTS)


class TestTheConstructorRefusesAnUnusableRate:
    """``1 / setpoint_hz`` runs inside a background thread, where nothing reports it."""

    @pytest.mark.parametrize("bad", [0, -1, 2.5, float("nan"), True, None, "20"])
    def test_a_rate_the_repeater_cannot_use_is_refused_at_construction(self, bad: object) -> None:
        with pytest.raises(ValueError, match="setpoint_hz"):
            module.CrazyflieDriver(setpoint_hz=bad)  # type: ignore[arg-type]

    def test_the_default_rate_is_accepted_and_reported(self) -> None:
        assert module.CrazyflieDriver().setpoint_hz == module.DEFAULT_SETPOINT_HZ


class TestThePolicyPathsRefuseRatherThanPretend:
    """No aerial policy provider exists here, and the refusal says what to use."""

    def test_start_task_and_run_policy_name_the_verbs_that_do_work(self, connected) -> None:  # type: ignore[no-untyped-def]
        driver, _, _ = connected()
        for envelope in (driver.start_task("fly to the window"), driver.run_policy(object())):  # type: ignore[arg-type]
            assert envelope["status"] == "error"
            text = envelope["content"][0]["text"]
            for verb in ("send_action", "set_twist", "takeoff", "land"):
                assert verb in text, f"the refusal must point at {verb}"
