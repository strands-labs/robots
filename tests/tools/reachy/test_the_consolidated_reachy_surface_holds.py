"""The consolidated Reachy tool surface holds - one table-driven suite.

Facts x verbs, in the shape the g1 family's consolidation locked (refs #3037,
#3070): the whole ``reachy_*`` surface is judged by rows in tables rather than
by a file per fact. Five fact families:

1. Every verb refuses an unusable ``driver`` handle with an envelope that
   names the verb, the parameter and the received type - never an exception.
2. Every verb makes exactly one call on a wired handle, through exactly the
   accessor its table row names, and returns that envelope object verbatim.
3. A data parameter a verb gates itself (the driver cannot see a missing
   ``emotion``) is refused with prose naming the parameter.
4. The package surface holds: every ``@tool`` the two verb modules define is
   lazily importable from the package - the drift the g1 family actually
   shipped once and now pins against.
5. No verb's description promises the head-body yaw coupling limit unless the
   action that verb sends can reach it. That limit needs both members of the
   pair in one action, so a verb sending one member is not refused by it - and
   the description is the only thing the model driving the verb reads.

Plus the new ``ReachyDriver`` accessor gates (connected-first, admitted sets,
URL-safe move names) exercised on a bare instance with no daemon, and the move
catalogue's wire shape, which is an array rather than a dict and so cannot be
read the way every other endpoint's body is.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

import strands_robots.drivers.reachy as reachy_driver_module
import strands_robots.tools.reachy as reachy_package
from strands_robots.drivers.reachy import ReachyDriver
from strands_robots.tools.reachy import reachy_actions, reachy_reads

ALL_VERBS: dict[str, Any] = {
    **{name: getattr(reachy_actions, name) for name in reachy_actions._ACTIONS},
    **{name: getattr(reachy_reads, name) for name in reachy_reads._READS},
}

#: verb -> the accessor its table row says it reads the handle through.
VERB_ACCESSOR: dict[str, str] = {
    **{name: row[0] for name, row in reachy_actions._ACTIONS.items()},
    **{name: row[0] for name, row in reachy_reads._READS.items()},
}

#: The arguments that carry each verb past its own data gates, so the call
#: reaches the driver accessor.
VERB_ARGS: dict[str, dict[str, Any]] = {
    "reachy_look": {"pitch": 10.0, "yaw": -5.0},
    "reachy_antennas": {"right": 30.0, "left": -30.0},
    "reachy_body_turn": {"yaw": 45.0},
    "reachy_home": {},
    "reachy_stop": {},
    "reachy_wake": {},
    "reachy_express": {"emotion": "happy"},
    "reachy_motors": {"mode": "enabled"},
    "reachy_play_sound": {"sound_file": "wake_up.wav"},
    "reachy_volume": {"level": 40},
    "reachy_camera": {},
    "reachy_look_at": {"u": 320, "v": 240},
    "reachy_get_state": {},
    "reachy_list_emotions": {},
}


class _RecordingHandle:
    """A handle that answers any accessor and records the calls it served."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.envelope = {"status": "success", "content": [{"json": {"from": "stub"}}]}

    def __getattr__(self, name: str) -> Any:
        def _call(*args: Any, **kwargs: Any) -> dict[str, Any]:
            self.calls.append((name, args, kwargs))
            return self.envelope

        return _call


class _AccessorIsData:
    """Carries every accessor name as *data* - a cache dump, not a driver."""

    def __init__(self) -> None:
        for accessor in set(VERB_ACCESSOR.values()):
            setattr(self, accessor, "a string, not a callable")


def _bare_driver(connected: bool = True) -> ReachyDriver:
    """A ReachyDriver with no daemon: caches empty, link absent."""
    driver = ReachyDriver.__new__(ReachyDriver)
    driver._connected = connected
    driver._cache_lock = threading.Lock()
    driver._head_yaw_target = None
    driver._joints = None
    driver._pose = None
    driver._imu = None
    driver._battery = None
    driver._link = None
    driver._loop = None
    driver._host = "localhost"
    driver._api_port = 8080
    return driver


class TestEveryVerbRefusesAWrongHandle:
    """Fact family 1: the live-handle judgement, uniform across the surface."""

    @pytest.mark.parametrize("verb", sorted(ALL_VERBS))
    def test_a_none_driver_is_refused_naming_the_verb_and_parameter(self, verb: str) -> None:
        result = ALL_VERBS[verb](driver=None)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith(f"{verb}: ")
        assert "`driver` is required" in text

    @pytest.mark.parametrize("verb", sorted(ALL_VERBS))
    def test_a_string_handle_is_refused_naming_the_received_type(self, verb: str) -> None:
        result = ALL_VERBS[verb](driver="reachy-mini")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith(f"{verb}: ")
        assert "'str'" in text

    @pytest.mark.parametrize("verb", sorted(ALL_VERBS))
    def test_an_accessor_present_as_data_is_still_refused(self, verb: str) -> None:
        result = ALL_VERBS[verb](driver=_AccessorIsData())
        assert result["status"] == "error"
        assert result["content"][0]["text"].startswith(f"{verb}: ")


class TestEveryVerbMakesExactlyOneDriverCall:
    """Fact family 2: one call, through the table's accessor, envelope verbatim."""

    @pytest.mark.parametrize("verb", sorted(ALL_VERBS))
    def test_the_tables_accessor_serves_the_call_and_the_envelope_returns_verbatim(self, verb: str) -> None:
        handle = _RecordingHandle()
        result = ALL_VERBS[verb](driver=handle, **VERB_ARGS[verb])
        assert result is handle.envelope
        assert len(handle.calls) == 1
        assert handle.calls[0][0] == VERB_ACCESSOR[verb]

    @pytest.mark.parametrize("sleep,accessor", [(False, "wake_up"), (True, "goto_sleep")])
    def test_reachy_wake_commands_the_motion_its_flag_names(self, sleep: bool, accessor: str) -> None:
        handle = _RecordingHandle()
        reachy_actions.reachy_wake(driver=handle, sleep=sleep)
        assert handle.calls == [(accessor, (), {})]

    def test_reachy_look_omits_axes_the_caller_left_alone(self) -> None:
        handle = _RecordingHandle()
        reachy_actions.reachy_look(driver=handle, pitch=15.0)
        (accessor, args, _) = handle.calls[0]
        action = args[0]
        assert accessor == "send_action"
        assert action["head_pitch"] == 15.0
        assert "body_yaw" not in action
        assert "antenna_left" not in action and "antenna_right" not in action

    def test_reachy_home_sends_every_axis_at_zero(self) -> None:
        handle = _RecordingHandle()
        reachy_actions.reachy_home(driver=handle)
        action = handle.calls[0][1][0]
        assert set(action) == {
            "head_pitch",
            "head_roll",
            "head_yaw",
            "head_x",
            "head_y",
            "head_z",
            "body_yaw",
            "antenna_left",
            "antenna_right",
        }
        assert all(value == 0.0 for value in action.values())


#: (verb, kwargs, fragment the refusal must contain) - the data gates the
#: verbs own because the driver cannot judge an absent value.
PARAM_REFUSALS: list[tuple[str, dict[str, Any], str]] = [
    ("reachy_express", {}, "`emotion` is required"),
    ("reachy_motors", {}, "`mode` is required"),
    ("reachy_play_sound", {}, "`sound_file` is required"),
    ("reachy_volume", {}, "`level` is required"),
    ("reachy_volume", {"level": True}, "got 'bool'"),
    ("reachy_volume", {"level": 101}, "within 0-100"),
    ("reachy_volume", {"level": -1}, "within 0-100"),
    ("reachy_look_at", {"v": 4}, "`u` is required"),
    ("reachy_look_at", {"u": 4}, "`v` is required"),
    ("reachy_look_at", {"u": 4.5, "v": 4}, "got 'float'"),
    ("reachy_look_at", {"u": True, "v": 4}, "got 'bool'"),
    ("reachy_wake", {"sleep": "false"}, "sleep must be a boolean"),
    ("reachy_wake", {"sleep": 1}, "sleep must be a boolean"),
]

#: Spellings a caller reaches for to opt *out* of sleep. Every one is truthy, so
#: a ``sleep`` read by truthiness commands go-to-sleep for each of them - the
#: opposite physical motion of the one asked for.
TRUTHY_SPELLINGS_OF_OFF: list[Any] = ["false", "no", "off", "0", "False"]


class TestTheVerbsOwnDataGates:
    """Fact family 3: a bad data parameter is refused naming the parameter."""

    @pytest.mark.parametrize(
        "verb,kwargs,fragment", PARAM_REFUSALS, ids=[f"{v}-{f[:20]}" for v, _, f in PARAM_REFUSALS]
    )
    def test_the_refusal_names_the_parameter(self, verb: str, kwargs: dict[str, Any], fragment: str) -> None:
        handle = _RecordingHandle()
        result = ALL_VERBS[verb](driver=handle, **kwargs)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith(f"{verb}: ")
        assert fragment in text
        assert handle.calls == []  # the refusal happened before any driver call


class TestTheWakeFlagSelectsAPostureRatherThanScalingAQuantity:
    """Fact family 3b: ``sleep`` picks one of two physical motions.

    It is therefore checked against
    :func:`~strands_robots.utils.boolean_flag_error` rather than read by
    truthiness. :class:`TestTheVerbsOwnDataGates` already grades the refusal's
    wording from :data:`PARAM_REFUSALS`; what these rows add is the *motion*,
    which is the part a wrong answer moves.
    """

    @pytest.mark.parametrize("spelling", TRUTHY_SPELLINGS_OF_OFF)
    def test_a_truthy_spelling_of_off_commands_nothing(self, spelling: Any) -> None:
        handle = _RecordingHandle()
        result = reachy_actions.reachy_wake(driver=handle, sleep=spelling)
        assert result["status"] == "error"
        # The point of the row: go-to-sleep was not commanded for a caller
        # spelling "do not go to sleep".
        assert handle.calls == []

    def test_the_flag_is_judged_before_the_accessor_it_selects(self) -> None:
        """A handle that can wake but not sleep still serves a request to wake.

        ``accessor`` is derived from ``sleep``, so a flag read by truthiness
        also decides which accessor the handle gate requires to be callable: a
        misread ``'false'`` made this handle unusable for the very motion it
        can perform.
        """

        class _WakeOnlyHandle:
            def wake_up(self) -> dict[str, Any]:
                return {"status": "success", "content": [{"text": "wake_up"}]}

        assert reachy_actions.reachy_wake(driver=_WakeOnlyHandle(), sleep=False)["status"] == "success"
        # Read from the table rather than written inline: the value is off-type
        # on purpose - that is the property under test - and the table's ``Any``
        # keeps the annotation mypy would enforce from hiding what a live caller
        # can still supply.
        refusal = reachy_actions.reachy_wake(driver=_WakeOnlyHandle(), sleep=TRUTHY_SPELLINGS_OF_OFF[0])
        assert refusal["status"] == "error"
        # The flag is named, not the handle - the handle was never the problem.
        assert "sleep must be a boolean" in refusal["content"][0]["text"]
        assert "does not expose" not in refusal["content"][0]["text"]


class TestThePackageSurfaceHolds:
    """Fact family 4: the lazy surface and the modules cannot drift apart."""

    def test_every_lazy_name_resolves(self) -> None:
        for name in reachy_package._LAZY_IMPORTS:
            assert getattr(reachy_package, name) is not None

    def test_every_tool_the_modules_define_is_on_the_package(self) -> None:
        defined = set(reachy_actions._ACTIONS) | set(reachy_reads._READS)
        assert defined == set(reachy_package._LAZY_IMPORTS)

    def test_the_tables_cover_every_tool_decorated_function(self) -> None:
        for module, table in ((reachy_actions, reachy_actions._ACTIONS), (reachy_reads, reachy_reads._READS)):
            tools = {
                name for name in dir(module) if not name.startswith("_") and hasattr(getattr(module, name), "tool_spec")
            }
            assert tools == set(table)


def _yaw_keys_one_call_carries(verb: str, kwargs: dict[str, Any]) -> set[str]:
    """The members of the yaw pair ``verb`` puts in one action, from a live call.

    Derived by calling the verb rather than read from a table, so a verb that
    starts or stops sending a member is graded without this file being edited.
    """
    handle = _RecordingHandle()
    ALL_VERBS[verb](driver=handle, **kwargs)
    for accessor, args, _kwargs in handle.calls:
        if accessor == "send_action" and args and isinstance(args[0], dict):
            return {"head_yaw", "body_yaw"} & set(args[0])
    return set()


class TestNoVerbPromisesACouplingLimitItsOwnActionCannotReach:
    """Fact family 5: a verb quoting the twist limit states the terms it applies on.

    A tool description is the only thing the model driving the verb reads, so a
    limit promised there is a limit the model plans against - and a promise of a
    refusal that does not happen is worse than no promise, because it moves a
    real robot. Which terms a verb owes depends on which member it can send
    alone, and the two are not the same obligation:

    * A lone ``head_yaw`` is not judged against a counterpart at all - the
      daemon serves it by turning the body under the head - so a verb that can
      omit ``body_yaw`` must say the limit applies only when both are sent.
    * A lone ``body_yaw`` *is* judged, against the head yaw the driver last
      commanded, so a verb that sends it alone must name that counterpart rather
      than claim an exemption it does not have.

    Derived from the tool specs and from what each verb actually sends, so
    re-adding an unqualified sentence - or keeping a stale exemption - fails here
    rather than shipping.
    """

    #: Either phrasing states the condition on a path where the limit is not
    #: applied. The wording stays the author's; only the presence of a stated
    #: condition is graded.
    SCOPE_PHRASES: tuple[str, ...] = ("only when", "not checked")

    #: What a verb sending a lone ``body_yaw`` must name instead: the head yaw
    #: the limit is measured against, since for that member it is measured.
    COUNTERPART_PHRASES: tuple[str, ...] = ("yaw target", "head's own yaw")

    @pytest.mark.parametrize("verb", sorted(reachy_actions._ACTIONS))
    def test_a_verb_that_can_send_one_member_states_the_condition(self, verb: str) -> None:
        limit = f"{reachy_package.HEAD_BODY_YAW_DELTA_LIMIT_DEG:g} deg"
        description = ALL_VERBS[verb].tool_spec["description"]
        if limit not in description:
            return  # says nothing about the coupling, so it promises nothing
        yaw_keys = _yaw_keys_one_call_carries(verb, VERB_ARGS[verb])
        if yaw_keys == {"head_yaw", "body_yaw"}:
            return  # every call carries the pair, so the limit does apply
        if yaw_keys == {"body_yaw"}:
            assert any(phrase in description for phrase in self.COUNTERPART_PHRASES), (
                f"{verb} sends body_yaw alone, where the {limit} coupling limit is applied "
                "against the head yaw target, and quotes the limit without naming it"
            )
            return
        assert any(phrase in description for phrase in self.SCOPE_PHRASES), (
            f"{verb} sends {sorted(yaw_keys) or 'no yaw key'} in one action and quotes the "
            f"{limit} coupling limit without stating when it applies"
        )

    def test_the_two_obligations_land_on_different_verbs(self) -> None:
        """The premise the branch above rests on, measured from the verbs.

        Without this, a single obligation applied to both would pass by accident
        if the two verbs happened to share a phrase.
        """
        lone = {
            verb: _yaw_keys_one_call_carries(verb, VERB_ARGS[verb])
            for verb in reachy_actions._ACTIONS
            if len(_yaw_keys_one_call_carries(verb, VERB_ARGS[verb])) == 1
        }
        assert lone == {"reachy_look": {"head_yaw"}, "reachy_body_turn": {"body_yaw"}}

    def test_the_two_verbs_that_can_send_a_lone_member_are_the_documented_pair(self) -> None:
        """The premise the rows above rest on, measured from the verbs."""
        lone = {verb for verb in reachy_actions._ACTIONS if len(_yaw_keys_one_call_carries(verb, VERB_ARGS[verb])) == 1}
        assert lone == {"reachy_look", "reachy_body_turn"}

    def test_passing_body_yaw_puts_the_pair_in_one_action(self) -> None:
        """``reachy_look``'s description offers this remedy, so it must exist."""
        assert _yaw_keys_one_call_carries("reachy_look", {"yaw": 80.0, "body_yaw": 0.0}) == {
            "head_yaw",
            "body_yaw",
        }


#: (method invocation, fragment) rows for the driver's own gates, judged on a
#: bare instance - no daemon, no link.
DRIVER_GATES: list[tuple[str, dict[str, Any], str]] = [
    ("play_move", {"move_name": "happy"}, "not connected"),
    ("list_moves", {}, "not connected"),
    ("wake_up", {}, "not connected"),
    ("goto_sleep", {}, "not connected"),
    ("set_motors", {"mode": "enabled"}, "not connected"),
    ("state_snapshot", {}, "not connected"),
]


class TestTheDriverAccessorGates:
    """The new ReachyDriver accessors refuse in envelopes, connected-first."""

    @pytest.mark.parametrize("method,kwargs,fragment", DRIVER_GATES, ids=[m for m, _, _ in DRIVER_GATES])
    def test_disconnected_is_refused(self, method: str, kwargs: dict[str, Any], fragment: str) -> None:
        driver = _bare_driver(connected=False)
        result = getattr(driver, method)(**kwargs)
        assert result["status"] == "error"
        assert fragment in result["content"][0]["text"]

    @pytest.mark.parametrize(
        "kwargs,fragment",
        [
            ({"move_name": "../evil"}, "invalid move_name"),
            ({"move_name": ""}, "invalid move_name"),
            ({"move_name": "ok", "library": "gestures"}, "unknown library"),
        ],
        ids=["path-escape", "empty-name", "unknown-library"],
    )
    def test_play_move_refuses_bad_input_before_any_request(self, kwargs: dict[str, Any], fragment: str) -> None:
        result = _bare_driver().play_move(**kwargs)
        assert result["status"] == "error"
        assert fragment in result["content"][0]["text"]

    def test_set_motors_refuses_the_sdk_only_mode_by_name(self) -> None:
        result = _bare_driver().set_motors("gravity_compensation")
        assert result["status"] == "error"
        assert "gravity_compensation" in result["content"][0]["text"]

    def test_set_motors_without_a_link_reports_the_link(self) -> None:
        result = _bare_driver().set_motors("disabled")
        assert result["status"] == "error"
        assert "link is not running" in result["content"][0]["text"]

    def test_state_snapshot_returns_cache_copies(self) -> None:
        driver = _bare_driver()
        driver._joints = {"body_yaw": 0.5}
        driver._battery = {"pct": 88}
        payload = driver.state_snapshot()["content"][0]["json"]
        assert payload["joints"] == {"body_yaw": 0.5}
        assert payload["battery"] == {"pct": 88}
        assert payload["imu"] is None and payload["pose"] is None
        payload["joints"]["body_yaw"] = 99  # a caller's mutation must not reach the cache
        assert driver._joints["body_yaw"] == 0.5


class _FakeTransport:
    """A transport double that answers ``api`` with one canned body."""

    def __init__(self, body: Any) -> None:
        self.body = body
        self.paths: list[str] = []

    def api(self, host: str, port: int, path: str, method: str = "GET", data: Any = None) -> Any:
        self.paths.append(path)
        return self.body


class TestTheMoveCatalogueIsReadAsAnArray:
    """Regression: the daemon's catalogue endpoint answers a JSON array.

    ``GET /api/move/recorded-move-datasets/list/{dataset}`` is declared
    ``-> list[str]`` by the daemon (pollen-robotics/reachy_mini,
    ``daemon/app/routers/move.py::list_recorded_move_dataset``) and
    ``reachy_transport.api`` returns the decoded body unreshaped, so on a
    *successful* read ``list_moves`` holds a ``list``. Reading ``.get("error")``
    off it raised ``AttributeError`` out through ``reachy_list_emotions``,
    breaking both halves of the surface's standing contract at once - an
    envelope back, and never an exception. The gates above could not see it
    because ``_RecordingHandle`` never returns a list, so these rows drive the
    transport seam with the shapes the daemon actually produces.
    """

    @staticmethod
    def _driver_answering(monkeypatch: pytest.MonkeyPatch, body: Any) -> tuple[ReachyDriver, _FakeTransport]:
        transport = _FakeTransport(body)
        monkeypatch.setattr(reachy_driver_module, "_resolve_transport", lambda: transport)
        return _bare_driver(), transport

    def test_a_catalogue_read_returns_the_array_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, transport = self._driver_answering(monkeypatch, ["happy", "sad", "curious"])
        result = driver.list_moves("emotions")
        assert result["status"] == "success"
        assert result["content"][0]["json"] == {
            "library": "emotions",
            "moves": ["happy", "sad", "curious"],
        }
        assert transport.paths == ["/api/move/recorded-move-datasets/list/pollen-robotics/reachy-mini-emotions-library"]

    def test_an_empty_catalogue_is_a_success_not_a_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _ = self._driver_answering(monkeypatch, [])
        result = driver.list_moves("dances")
        assert result["status"] == "success"
        assert result["content"][0]["json"]["moves"] == []

    def test_each_library_reads_its_own_dataset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, transport = self._driver_answering(monkeypatch, [])
        driver.list_moves("dances")
        assert transport.paths == ["/api/move/recorded-move-datasets/list/pollen-robotics/reachy-mini-dances-library"]

    def test_an_error_body_is_still_refused_naming_the_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _ = self._driver_answering(monkeypatch, {"error": "daemon said no"})
        result = driver.list_moves()
        assert result["status"] == "error"
        assert "daemon said no" in result["content"][0]["text"]

    def test_an_unexpected_dict_is_refused_rather_than_served_as_a_catalogue(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver, _ = self._driver_answering(monkeypatch, {"unexpected": "shape"})
        result = driver.list_moves()
        assert result["status"] == "error"
        assert "list_moves: daemon refused" in result["content"][0]["text"]

    def test_the_read_verb_carries_the_array_out_without_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _ = self._driver_answering(monkeypatch, ["happy"])
        result = reachy_reads.reachy_list_emotions(driver=driver)
        assert result["status"] == "success"
        assert result["content"][0]["json"]["moves"] == ["happy"]
