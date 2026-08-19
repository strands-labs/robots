"""A teleop device selector is read by membership, never by truthiness.

``teleoperate(names=)`` and ``detach_teleop(name=)`` both pick which of the
already-attached streams a call operates on, and both document ``None`` as
"every attached device". Read by truthiness, every other falsy value took that
same branch, so a selection that named *nothing* was widened to *everything* -
the opposite of what the caller asked for, reported as success:

* ``teleoperate(names=[])`` - what a filter that matched nothing produces -
  connected and drove every attached device.
* ``detach_teleop("")`` removed the whole attached set and, with a session
  running, ended it, because ``detach_teleop`` stops the loop once nothing is
  left to drive.

Two surfaces in the same module already read a selector the way these did not:
``teleoperate`` itself validates ``duration`` by membership so "a falsy-but-
supplied value must not read as absent", and the render path resolves its
``cameras`` subset by membership, where an empty selection resolves to no
camera rather than to every one.

The list shape goes through the shared name-list domain, which is where the
other unhonorable spellings of this selector live: a single name as a bare
string is iterable per character, a repeated name polls one device twice per
tick, and a one-shot iterator is consumed by the membership check that resolves
unknown names, leaving the loop nothing to poll.
"""

from __future__ import annotations

import time

import pytest

from tests.test_teleop import FakeHost, FakeTeleop


def _host_with_two_devices() -> FakeHost:
    """A host with a leader and a gamepad, neither connected yet."""
    host = FakeHost()
    host.attach_teleop(FakeTeleop({"shoulder.pos": 1.0}, name="so101_leader"), name="leader")
    host.attach_teleop(FakeTeleop({"btn.x": 2.0}, name="gamepad"), name="pad")
    return host


def _driven(host: FakeHost) -> dict[str, int]:
    """How many times each attached device was polled for an action."""
    return {name: att.device.get_action_calls for name, att in host._teleops.items()}


def _connected(host: FakeHost) -> dict[str, int]:
    """How many times each attached device was connected."""
    return {name: att.device.connect_calls for name, att in host._teleops.items()}


class TestAnEmptySelectionSelectsNothing:
    """``names=[]`` names no device, so it cannot resolve to every device."""

    def test_an_empty_selection_does_not_drive_an_unselected_device(self) -> None:
        host = _host_with_two_devices()

        result = host.teleoperate(names=[], block=True, duration=0.15, hz=50.0)

        polled = _driven(host)
        if polled["pad"]:
            pytest.fail(
                f"teleoperate(names=[]) selected no device and then polled "
                f"{sum(1 for v in polled.values() if v)} of them {polled}, including 'pad', "
                f"which the call did not name; it connected {_connected(host)} and reported "
                f"{result['status']!r}."
            )
        assert result["status"] == "error"

    def test_an_empty_selection_is_not_the_same_call_as_no_selection(self) -> None:
        """The two must not be interchangeable: they ask for opposite things."""
        empty = _host_with_two_devices()
        empty_result = empty.teleoperate(names=[], block=True, duration=0.15, hz=50.0)

        absent = _host_with_two_devices()
        absent_result = absent.teleoperate(names=None, block=True, duration=0.15, hz=50.0)

        assert absent_result["status"] == "success", "premise: names=None drives every device"
        assert all(v > 0 for v in _driven(absent).values()), "premise: names=None polled both"

        assert empty_result["status"] != absent_result["status"], (
            f"names=[] and names=None both reported {empty_result['status']!r}; an empty "
            f"selection resolved to the same set as no selection ({_driven(empty)})"
        )

    def test_the_empty_refusal_names_the_spelling_that_drives_everything(self) -> None:
        """Follow the remedy the refusal gives and it must do what it promises."""
        host = _host_with_two_devices()

        refusal = host.teleoperate(names=[], block=True, duration=0.15, hz=50.0)
        assert refusal["status"] == "error"
        text = refusal["content"][0]["text"]
        assert "names=None" in text, f"refusal does not name the all-devices spelling: {text!r}"
        for attached in ("leader", "pad"):
            assert attached in text, f"refusal does not list the attached devices: {text!r}"

        # Apply the remedy verbatim on a fresh host.
        remedied = _host_with_two_devices()
        result = remedied.teleoperate(names=None, block=True, duration=0.15, hz=50.0)
        assert result["status"] == "success"
        assert all(v > 0 for v in _driven(remedied).values()), (
            f"the remedy the refusal advertised did not drive every device: {_driven(remedied)}"
        )

    def test_an_empty_selection_is_refused_before_any_device_is_connected(self) -> None:
        """A refused selection must not have touched hardware on its way out."""
        host = _host_with_two_devices()

        host.teleoperate(names=[], block=True, duration=0.15, hz=50.0)

        assert _connected(host) == {"leader": 0, "pad": 0}
        assert all(not att.device.is_connected for att in host._teleops.values())


class TestTheListShapeGoesThroughTheSharedDomain:
    """Spellings of ``names`` that cannot be honored as written are refused."""

    @pytest.mark.parametrize(
        ("names", "expected_fragment"),
        [
            pytest.param("leader", "not a single string", id="bare-string"),
            pytest.param(["leader", "leader"], "must not repeat a name", id="repeated-name"),
            pytest.param({"leader": 1}, "not a mapping", id="mapping"),
        ],
    )
    def test_an_unhonorable_shape_is_refused_naming_the_value(self, names: object, expected_fragment: str) -> None:
        host = _host_with_two_devices()

        result = host.teleoperate(names=names, block=True, duration=0.15, hz=50.0)  # type: ignore[arg-type]

        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert expected_fragment in text, f"refusal does not diagnose the shape: {text!r}"
        assert _driven(host) == {"leader": 0, "pad": 0}, f"a refused shape still drove devices: {_driven(host)}"

    def test_a_repeated_name_does_not_poll_one_device_twice_per_tick(self) -> None:
        host = _host_with_two_devices()

        result = host.teleoperate(names=["leader", "leader"], block=True, duration=0.2, hz=50.0)

        frames = result["content"][-1].get("json", {}).get("frames", 0)
        polled = _driven(host)["leader"]
        assert result["status"] == "error", (
            f"a repeated name ran a session that polled 'leader' {polled} times across {frames} frames"
        )

    def test_a_one_shot_iterator_is_refused_rather_than_consumed(self) -> None:
        """Consumed by the unknown-name check, an iterator left the loop nothing."""
        host = _host_with_two_devices()

        result = host.teleoperate(names=iter(["leader"]), block=True, duration=0.15, hz=50.0)  # type: ignore[arg-type]

        telemetry = result["content"][-1].get("json", {})
        assert result["status"] == "error", (
            f"an iterator ran a session reporting {result['status']!r} with "
            f"{telemetry.get('frames')} frames after polling {_driven(host)}"
        )


class TestOnlyNoneDetachesEveryDevice:
    """``detach_teleop`` selects by membership too."""

    def test_an_empty_name_does_not_detach_a_device_it_does_not_name(self) -> None:
        host = _host_with_two_devices()

        result = host.detach_teleop("")

        assert sorted(host._teleops) == ["leader", "pad"], (
            f"detach_teleop('') detached {result['content'][0]['text']!r}; remaining: {sorted(host._teleops)}"
        )
        assert result["status"] == "error"
        assert "''" in result["content"][0]["text"]

    def test_an_empty_name_does_not_end_a_running_session(self) -> None:
        host = _host_with_two_devices()
        started = host.teleoperate(hz=50.0)
        assert started["status"] == "success", "premise: session started"
        time.sleep(0.1)
        assert host._teleop_running, "premise: the loop is running"
        try:
            frames_before = host._teleop_frames

            host.detach_teleop("")
            time.sleep(0.1)

            assert host._teleop_running, (
                "detach_teleop('') stopped a running session by detaching every device "
                f"(frames froze at {frames_before})"
            )
            assert host._teleop_frames > frames_before, "the loop stopped producing frames"
        finally:
            host.stop_teleoperate()


class TestTheDocumentedSelectionsAreUnchanged:
    """Controls: the spellings that already worked keep working."""

    def test_no_selection_still_drives_every_attached_device(self) -> None:
        host = _host_with_two_devices()

        result = host.teleoperate(names=None, block=True, duration=0.15, hz=50.0)

        assert result["status"] == "success"
        assert all(v > 0 for v in _driven(host).values()), _driven(host)

    def test_a_named_subset_still_drives_only_its_own_devices(self) -> None:
        host = _host_with_two_devices()

        result = host.teleoperate(names=["leader"], block=True, duration=0.15, hz=50.0)

        assert result["status"] == "success"
        polled = _driven(host)
        assert polled["leader"] > 0 and polled["pad"] == 0, polled
        assert {k for action, _ in host.sent for k in action} == {"shoulder.pos"}

    def test_an_unknown_name_in_a_usable_list_still_reports_it(self) -> None:
        host = _host_with_two_devices()

        result = host.teleoperate(names=["ghost"], block=True, duration=0.15, hz=50.0)

        assert result["status"] == "error"
        assert "Unknown teleop name(s): ['ghost']" in result["content"][0]["text"]

    def test_detach_with_no_name_still_detaches_every_device(self) -> None:
        host = _host_with_two_devices()

        result = host.detach_teleop()

        assert result["status"] == "success"
        assert host._teleops == {}

    def test_detach_of_an_unattached_name_still_changes_nothing(self) -> None:
        host = _host_with_two_devices()

        result = host.detach_teleop("nosuch")

        assert result["status"] == "error"
        assert sorted(host._teleops) == ["leader", "pad"]

    def test_an_empty_selection_on_a_host_with_no_devices_reports_the_real_problem(self) -> None:
        """Nothing attached outranks an empty selection: it is the fixable one."""
        host = FakeHost()

        result = host.teleoperate(names=[], block=True, duration=0.15, hz=50.0)

        assert result["status"] == "error"
        assert "No teleoperators attached" in result["content"][0]["text"]
