"""Behavior tests for the ``use_rtps`` tool and the participant it publishes through.

``use_rtps`` is the agent envelope over a pure-RTPS ROS 2 participant on
cyclonedds (:mod:`strands_robots.rtps.participant`, which
:class:`~strands_robots.mesh.rtps_robot.RtpsRobot` publishes through as well).
These tests are hermetic: they neither require nor reject an installed
cyclonedds / ROS 2 -- the backend's ``available`` probe and its writer/reader
factories are monkeypatched, so every action-dispatch branch, the agent-input
validation, the no-backend error path, and the structured error-return contract
are exercised middleware-free on any machine.

The tool is driven end to end throughout: the dispatch, the DDS-entity caching
and the pure sample helpers all belong to the participant, and reaching them
through the tool grades the envelope and the participant together rather than
one of them twice.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

import strands_robots.rtps.participant as participant_mod
import strands_robots.tools.use_rtps as rtps_mod

use_rtps = rtps_mod.use_rtps


# Module-level fake IDL dataclasses so typing.get_type_hints can resolve nested
# field types against module globals (mirrors the real module-level IDL bundle).
@dataclasses.dataclass
class _Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclasses.dataclass
class _Twist:
    linear: _Vec3 = dataclasses.field(default_factory=_Vec3)
    angular: _Vec3 = dataclasses.field(default_factory=_Vec3)


@dataclasses.dataclass
class _FlatTwist:
    linear: float = 0.0


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _ascii_only(result: dict[str, Any]) -> None:
    assert _texts(result).isascii(), f"non-ASCII in output: {_texts(result)!r}"


class _FakeWriter:
    def __init__(self) -> None:
        self.written: list[Any] = []

    def write(self, sample: Any) -> None:
        self.written.append(sample)


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> _FakeWriter:
    """Patch the backend to be available with a recording writer; no real DDS."""
    writer = _FakeWriter()
    monkeypatch.setattr(participant_mod._backend, "available", lambda: True)
    monkeypatch.setattr(participant_mod._backend, "writer", lambda topic, type: writer)
    # publish sleeps for settle/rate; make it instant.
    monkeypatch.setattr(participant_mod.time, "sleep", lambda *_: None)
    return writer


# Validation ----------------------------------------------------------------


@pytest.mark.parametrize("bad", ["cmd_vel", "/bad name", "/x;y", "../etc"])
def test_invalid_topic_rejected(bad: str) -> None:
    result = use_rtps(action="publish", topic=bad, type="geometry_msgs/msg/Twist")
    assert result["status"] == "error"
    assert "invalid topic" in _texts(result)
    _ascii_only(result)


def test_invalid_type_rejected() -> None:
    result = use_rtps(action="publish", topic="/cmd_vel", type="not_a_type")
    assert result["status"] == "error"
    assert "invalid interface type" in _texts(result)


@pytest.mark.parametrize(
    ("kind_type", "generated"),
    [
        ("example_interfaces/srv/AddTwoInts", "example_interfaces::srv::dds_::AddTwoInts_Request_"),
        ("nav2_msgs/action/NavigateToPose", "nav2_msgs::action::dds_::NavigateToPose_Goal_"),
    ],
)
def test_service_and_action_types_are_refused_before_the_backend_probe(
    kind_type: str, generated: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # docs/rtps-integration.md, Type coverage: a pkg/srv/Name or pkg/action/Name
    # type is refused with the types ROS 2 does generate quoted. The refusal is
    # the mangling's own (dds_type_name), so it reads the same with or without
    # cyclonedds and never reaches the install hint or the IDL bundle lookup.
    monkeypatch.setattr(participant_mod._backend, "available", lambda: False)
    result = use_rtps(action="advertise", topic="/x", type=kind_type)
    assert result["status"] == "error"
    text = _texts(result)
    assert "invalid interface type" in text
    assert generated in text
    assert "cyclonedds is required" not in text
    _ascii_only(result)


# Status / no-backend -------------------------------------------------------


def test_status_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(participant_mod._backend, "available", lambda: True)
    result = use_rtps(action="status")
    assert result["status"] == "success"
    assert "cyclonedds" in _texts(result)
    _ascii_only(result)


def test_status_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(participant_mod._backend, "available", lambda: False)
    result = use_rtps(action="status")
    assert result["status"] == "success"
    assert "backend: none" in _texts(result)


def test_action_without_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(participant_mod._backend, "available", lambda: False)
    result = use_rtps(action="publish", topic="/cmd_vel", type="geometry_msgs/msg/Twist")
    assert result["status"] == "error"
    assert "cyclonedds" in _texts(result)
    _ascii_only(result)


# types ---------------------------------------------------------------------


def test_types_lists_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(participant_mod._backend, "available", lambda: True)
    # REGISTRY is imported inside the function from the idl module; patch there.
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "REGISTRY", {"geometry_msgs/msg/Twist": object})
    result = use_rtps(action="types")
    assert result["status"] == "success"
    assert "geometry_msgs/msg/Twist" in _texts(result)


# publish (the headline "act as a robot" path) ------------------------------


# A topic that is not a safety-critical command surface. The tests below exercise
# publish plumbing; aiming them at a drive topic would route them through the
# operator gate in :mod:`strands_robots._command_gate`, which is a different
# subject with its own suite.
_PLUMBING_TOPIC = "/demo/twist"


def test_publish_builds_and_writes_count(fake_backend: _FakeWriter, monkeypatch: pytest.MonkeyPatch) -> None:
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "get_type", lambda t: _Twist)
    monkeypatch.setattr(idl_mod, "REGISTRY", {"geometry_msgs/msg/Twist": _Twist})

    result = use_rtps(
        action="publish",
        topic=_PLUMBING_TOPIC,
        type="geometry_msgs/msg/Twist",
        fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}},
        count=3,
    )
    assert result["status"] == "success"
    assert f"published 3 message(s) to {_PLUMBING_TOPIC}" in _texts(result)
    assert len(fake_backend.written) == 3
    # Nested field dict was built into real dataclass instances, types intact.
    sent = fake_backend.written[0]
    assert sent.linear.x == 2.0
    assert sent.angular.z == 1.5


def test_publish_unknown_field_is_structured_error(fake_backend: _FakeWriter, monkeypatch: pytest.MonkeyPatch) -> None:
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "get_type", lambda t: _FlatTwist)
    result = use_rtps(
        action="publish",
        topic=_PLUMBING_TOPIC,
        type="geometry_msgs/msg/Twist",
        fields={"bogus": 1.0},
    )
    assert result["status"] == "error"
    assert "publish failed" in _texts(result)
    assert "unknown field" in _texts(result)


def test_publish_requires_topic_and_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(participant_mod._backend, "available", lambda: True)
    assert use_rtps(action="publish", topic="/cmd_vel")["status"] == "error"


# advertise / unknown -------------------------------------------------------


def test_advertise_creates_writer(fake_backend: _FakeWriter, monkeypatch: pytest.MonkeyPatch) -> None:
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "get_type", lambda t: object)
    result = use_rtps(action="advertise", topic="/turtle1/cmd_vel", type="geometry_msgs/msg/Twist")
    assert result["status"] == "success"
    assert "advertised /turtle1/cmd_vel" in _texts(result)


def test_unknown_action_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(participant_mod._backend, "available", lambda: True)
    result = use_rtps(action="warp_drive")
    assert result["status"] == "error"
    assert "unknown action" in _texts(result)


def test_unknown_action_is_named_before_the_backend_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd verb is refused by name on a box without cyclonedds, not answered by the install recipe."""
    monkeypatch.setattr(participant_mod._backend, "available", lambda: False)
    forwarded: list[str] = []

    def _participant(action: str, **_: Any) -> dict[str, Any]:
        forwarded.append(action)
        return {"status": "error", "content": [{"text": "participant reached"}]}

    monkeypatch.setattr(rtps_mod, "rtps_action", _participant)
    result = use_rtps(action="pubilsh")
    assert result["status"] == "error"
    assert "pubilsh" in _texts(result)
    assert "publish" in _texts(result) and "cyclonedds" not in _texts(result)
    assert forwarded == []


# subscribe / echo (the read-side participant path) -------------------------


class _FakeReader:
    """DataReader stub: yields a fixed batch per ``take`` call, then empties.

    Mirrors the cyclonedds ``DataReader.take(N=...)`` contract closely enough to
    drive the echo poll loop without a live DDS graph.
    """

    def __init__(self, batches: list[list[Any]]) -> None:
        self._batches = list(batches)

    def take(self, N: int) -> list[Any]:
        return self._batches.pop(0) if self._batches else []


@pytest.fixture
def with_reader(monkeypatch: pytest.MonkeyPatch):
    """Patch the backend available + reader factory; return a setter for batches."""
    monkeypatch.setattr(participant_mod._backend, "available", lambda: True)
    monkeypatch.setattr(participant_mod.time, "sleep", lambda *_: None)

    def _install(batches: list[list[Any]]) -> _FakeReader:
        reader = _FakeReader(batches)
        monkeypatch.setattr(participant_mod._backend, "reader", lambda topic, type: reader)
        return reader

    return _install


def test_subscribe_creates_reader(with_reader) -> None:
    with_reader([])
    result = use_rtps(action="subscribe", topic="/turtle1/cmd_vel", type="geometry_msgs/msg/Twist")
    assert result["status"] == "success"
    assert "subscribed to /turtle1/cmd_vel" in _texts(result)
    _ascii_only(result)


def test_echo_returns_samples_as_dicts(with_reader) -> None:
    # Two single-sample batches force the partial-take + poll-again branch.
    with_reader([[_Twist(linear=_Vec3(x=2.0))], [_Twist(angular=_Vec3(z=1.5))]])
    result = use_rtps(
        action="echo",
        topic="/turtle1/cmd_vel",
        type="geometry_msgs/msg/Twist",
        count=2,
        timeout=5.0,
    )
    assert result["status"] == "success"
    text = _texts(result)
    assert "echo /turtle1/cmd_vel" in text
    # Samples were recursively converted to nested plain dicts.
    payload = json.loads(text.split("):\n", 1)[1])
    assert payload[0]["linear"]["x"] == 2.0
    assert payload[1]["angular"]["z"] == 1.5
    _ascii_only(result)


def test_echo_times_out_with_empty_samples(with_reader) -> None:
    """An expired wait returns an empty sample list rather than an error.

    The budget is the smallest usable one rather than ``0.0``: a zero timeout is
    not a wait a caller can ask for (it is refused up front), and it also skipped
    the polling loop entirely, so the branch under test never ran.
    """
    with_reader([])  # reader never yields
    result = use_rtps(
        action="echo",
        topic="/cmd_vel",
        type="geometry_msgs/msg/Twist",
        count=1,
        timeout=0.01,  # one poll, then the deadline passes -> empty result
    )
    assert result["status"] == "success"
    text = _texts(result)
    assert json.loads(text.split("):\n", 1)[1]) == []


def test_echo_requires_topic_and_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(participant_mod._backend, "available", lambda: True)
    result = use_rtps(action="echo", topic="/cmd_vel")
    assert result["status"] == "error"
    assert "echo requires topic and type" in _texts(result)


def test_advertise_requires_topic_and_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(participant_mod._backend, "available", lambda: True)
    result = use_rtps(action="advertise", topic="/cmd_vel")
    assert result["status"] == "error"
    assert "advertise requires topic and type" in _texts(result)


# Pure helpers --------------------------------------------------------------


def test_sample_to_dict_handles_nested_lists_and_scalars() -> None:
    sample = _Twist(linear=_Vec3(x=1.0, y=2.0), angular=_Vec3(z=3.0))
    out = participant_mod._sample_to_dict([sample, 7])
    assert out[0]["linear"] == {"x": 1.0, "y": 2.0, "z": 0.0}
    assert out[0]["angular"]["z"] == 3.0
    assert out[1] == 7  # scalars pass through unchanged


def test_resolve_field_types_falls_back_when_hints_unresolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_cls: Any) -> dict[str, Any]:
        raise NameError("unresolved forward ref")

    monkeypatch.setattr(participant_mod.typing, "get_type_hints", _boom)
    resolved = participant_mod._resolve_field_types(_Vec3)
    # Falls back to the raw dataclasses Field.type (a string under future-annotations).
    assert set(resolved) == {"x", "y", "z"}


# Backend availability probe (have_cyclonedds gated) ------------------------


def test_backend_available_true_when_cyclonedds_present(monkeypatch: pytest.MonkeyPatch) -> None:
    # available() delegates to strands_robots.rtps.idl.have_cyclonedds(). Drive
    # the present-dependency branch deterministically rather than depending on
    # whether cyclonedds happens to be installed in the test environment.
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "have_cyclonedds", lambda: True)
    assert participant_mod._RtpsBackend().available() is True


def test_backend_available_false_without_cyclonedds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate cyclonedds being absent regardless of the test environment.
    # available() imports have_cyclonedds() from strands_robots.rtps.idl on each
    # call, so forcing that probe to report False exercises the documented
    # dependency-absent fallback (the tool degrades to a clear status message)
    # on a machine where cyclonedds IS installed too.
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "have_cyclonedds", lambda: False)
    assert participant_mod._RtpsBackend().available() is False


def test_backend_available_import_error_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fail(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "strands_robots.rtps.idl" or name.endswith("rtps.idl"):
            raise ImportError("simulated missing idl module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail)
    assert participant_mod._RtpsBackend().available() is False


# Backend DDS-entity caching contract --------------------------------------
#
# The action-dispatch tests above stub ``_backend.writer`` / ``_backend.reader``
# so they never reach the real get-or-create factories. These tests drive the
# real ``_RtpsBackend`` methods with fake ``cyclonedds`` submodules injected
# into ``sys.modules`` (overriding any real install), exercising the documented
# contract: one shared DomainParticipant, and writers/readers cached per
# (topic, type) so repeated calls reuse the same DDS entity.


@dataclasses.dataclass
class _EntityRecorder:
    """Records every DDS entity the backend constructs via the fake modules."""

    participants: list[Any] = dataclasses.field(default_factory=list)
    topics: list[Any] = dataclasses.field(default_factory=list)
    writers: list[Any] = dataclasses.field(default_factory=list)
    readers: list[Any] = dataclasses.field(default_factory=list)


@pytest.fixture
def rtps_entities(monkeypatch: pytest.MonkeyPatch) -> _EntityRecorder:
    """Inject fake ``cyclonedds`` submodules + IDL resolvers; record construction.

    Returns the recorder so a test can assert how many participants/topics/etc.
    the backend created. No real DDS or cyclonedds wheel is involved.
    """
    import sys
    import types

    rec = _EntityRecorder()

    class _FakeParticipant:
        def __init__(self) -> None:
            rec.participants.append(self)

    class _FakeTopic:
        def __init__(self, participant: Any, name: str, idl_cls: Any) -> None:
            self.participant = participant
            self.name = name
            self.idl_cls = idl_cls
            rec.topics.append(self)

    class _FakeDataWriter:
        def __init__(self, participant: Any, topic: Any) -> None:
            self.participant = participant
            self.topic = topic
            rec.writers.append(self)

    class _FakeDataReader:
        def __init__(self, participant: Any, topic: Any) -> None:
            self.participant = participant
            self.topic = topic
            rec.readers.append(self)

    def _mod(name: str, **attrs: Any) -> types.ModuleType:
        m = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(m, key, value)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    _mod("cyclonedds")
    _mod("cyclonedds.domain", DomainParticipant=_FakeParticipant)
    _mod("cyclonedds.topic", Topic=_FakeTopic)
    _mod("cyclonedds.pub", DataWriter=_FakeDataWriter)
    _mod("cyclonedds.sub", DataReader=_FakeDataReader)

    # The factories resolve the IDL class + mangle the topic name from the real
    # modules; pin them so the test is independent of the shipped IDL bundle.
    import strands_robots.rtps.idl as idl_mod
    import strands_robots.rtps.mangling as mangling_mod

    monkeypatch.setattr(idl_mod, "get_type", lambda ros_type: _Twist)
    monkeypatch.setattr(mangling_mod, "dds_topic_name", lambda topic: f"rt{topic}")
    return rec


def test_participant_created_once_and_shared(rtps_entities: _EntityRecorder) -> None:
    backend = participant_mod._RtpsBackend()
    first = backend._participant_obj()
    second = backend._participant_obj()
    assert first is second  # one shared presence on the graph


def test_writer_get_or_create_caches_per_topic_type(rtps_entities: _EntityRecorder) -> None:
    backend = participant_mod._RtpsBackend()

    w1 = backend.writer("/turtle1/cmd_vel", "geometry_msgs/msg/Twist")
    w1_again = backend.writer("/turtle1/cmd_vel", "geometry_msgs/msg/Twist")
    w2 = backend.writer("/other", "geometry_msgs/msg/Twist")

    # Same (topic, type) reuses the cached writer; a new key builds a new one.
    assert w1 is w1_again
    assert w2 is not w1
    assert len(rtps_entities.writers) == 2
    # One participant shared across both writers.
    assert len(rtps_entities.participants) == 1
    assert w1.participant is w2.participant
    # Topic was built with the mangled name and the resolved IDL class.
    assert w1.topic.name == "rt/turtle1/cmd_vel"
    assert w1.topic.idl_cls is _Twist


def test_reader_get_or_create_caches_per_topic_type(rtps_entities: _EntityRecorder) -> None:
    backend = participant_mod._RtpsBackend()

    r1 = backend.reader("/turtle1/cmd_vel", "geometry_msgs/msg/Twist")
    r1_again = backend.reader("/turtle1/cmd_vel", "geometry_msgs/msg/Twist")
    r2 = backend.reader("/other", "geometry_msgs/msg/Twist")

    assert r1 is r1_again
    assert r2 is not r1
    assert len(rtps_entities.readers) == 2
    assert len(rtps_entities.participants) == 1
    assert r1.topic.name == "rt/turtle1/cmd_vel"


def test_writer_and_reader_share_one_participant(rtps_entities: _EntityRecorder) -> None:
    backend = participant_mod._RtpsBackend()
    writer = backend.writer("/cmd_vel", "geometry_msgs/msg/Twist")
    reader = backend.reader("/cmd_vel", "geometry_msgs/msg/Twist")
    # A single DomainParticipant backs both directions of the participant.
    assert len(rtps_entities.participants) == 1
    assert writer.participant is reader.participant
