"""Gate row 1: a tool spec for every rpc of the device, its JSON schema generated from the signature."""

from __future__ import annotations

import json
from dataclasses import dataclass

from strands_robots import Robot
from strands_robots.tools import parameters, tool_spec


def _schema(spec: dict) -> dict:
    return spec["inputSchema"]["json"]


def test_robot_tool_spec_lists_every_verb_with_schemas_from_signatures():
    arm = Robot("so101")
    spec = tool_spec(arm)
    json.dumps(spec)
    schema = _schema(spec)
    assert schema["required"] == ["rpc"]
    assert set(schema["properties"]["rpc"]["enum"]) >= {"describe", "observe", "act", "run", "stop"}
    act = schema["properties"]["act"]
    assert act["type"] == "object"
    assert set(act["properties"]["action"]["properties"]) == set(arm.action_keys)
    run = schema["properties"]["run"]["properties"]
    assert run["policy"]["type"] == "string"
    assert run["steps"]["type"] == ["integer", "null"] and run["hz"]["type"] == ["number", "null"]
    assert "stop_when" not in run and "stop_event" not in run  # callables do not cross the wire
    assert spec["name"] == "robot_so101" and "shoulder_pan" in spec["description"]


def test_parameters_follow_the_signature():
    def wave(times: int, *, height: float = 0.1, side: str | None = None, fast: bool = False) -> None:
        """Wave the arm."""

    p = parameters(wave, joints=())
    assert p["required"] == ["times"]
    assert p["properties"]["times"]["type"] == "integer"
    assert p["properties"]["height"] == {"type": "number", "default": 0.1}
    assert p["properties"]["side"]["type"] == ["string", "null"]
    assert p["properties"]["fast"]["type"] == "boolean"


@dataclass(frozen=True)
class _Rpc:
    name: str


@dataclass(frozen=True)
class _Manifest:
    rpcs: tuple[_Rpc, ...]


class _Device:
    """A device whose manifest declares its rpcs; the tool is generated from that list alone."""

    id = "bench_1"
    action_keys = ("a", "b")

    def __init__(self, rpcs: tuple[str, ...]) -> None:
        self.manifest = _Manifest(tuple(_Rpc(n) for n in rpcs))

    def describe(self) -> dict:
        return {"id": self.id}

    def wave(self, times: int, *, height: float = 0.1) -> None:
        """Wave the arm."""


def test_a_manifest_change_changes_the_tool_with_no_tool_code_edit():
    before = _schema(tool_spec(_Device(("describe",))))
    after = _schema(tool_spec(_Device(("describe", "wave"))))
    assert before["properties"]["rpc"]["enum"] == ["describe"] and "wave" not in before["properties"]
    assert after["properties"]["rpc"]["enum"] == ["describe", "wave"]
    assert after["properties"]["wave"]["properties"]["times"]["type"] == "integer"
    assert "wave: Wave the arm." in after["properties"]["rpc"]["description"]


def test_robot_run_has_the_run_loops_own_signature_and_start_every_keyword_but_stop_event():
    import inspect

    from strands_robots.robot.run import loop

    run = inspect.signature(Robot.run).parameters
    assert run == inspect.signature(loop.run).parameters
    start = dict(inspect.signature(Robot.start).parameters)
    assert list(start) == [p for p in run if p != "stop_event"]


def test_the_tool_offers_exactly_the_manifest_rpcs_an_agent_may_call():
    arm = Robot("so101")
    declared = [r.name for r in arm.manifest.rpcs]
    for r in arm.manifest.rpcs:
        assert "robot" not in r.params["properties"] and "self" not in r.params["properties"], r
    enum = _schema(tool_spec(arm))["properties"]["rpc"]["enum"]
    assert enum == [n for n in declared if n not in ("resume", "calibrate")]  # the operator's, never the model's
    assert "estop" in enum and "hold" in enum


def test_nested_generics_and_unions_are_read_exactly():
    def f(
        x: dict[str, int | float],
        y: tuple[int, int] | None = None,
        z: list[float] | None = None,
        live: Member | str = "a",  # noqa: F821 - a type-checking-only name stays off the wire, str stays
    ) -> None:
        """F."""

    p = parameters(f, joints=())["properties"]
    assert p["x"] == {"type": "object", "additionalProperties": {"type": "integer"}}
    assert p["y"] == {"type": ["array", "null"], "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}
    assert p["z"]["items"] == {"type": "number"} and p["live"] == {"type": "string", "default": "a"}


def test_long_device_ids_never_collide_in_the_64_char_tool_name():
    from strands_robots.tools.spec import tool_name

    a, b = tool_name("robot", "x" * 70 + "_left"), tool_name("robot", "x" * 70 + "_right")
    assert len(a) <= 64 and len(b) <= 64 and a != b
    assert tool_name("robot", "arm-1") == "robot_arm_1"
