"""The ``use_unitree`` meta operations answer without a robot, DDS, or the SDK.

The universal dispatcher's discovery surface (``list_services`` /
``list_operations`` / ``describe_operation``) is the replacement for the
per-method lookup modules the consolidation removed (refs #2928): an agent
asks the dispatcher what exists instead of carrying one ``@tool`` per
constant table. That replacement only holds if discovery works on a machine
with no ``unitree_sdk2py`` import and no CycloneDDS - this file pins that.

Also pins the SDK-load-hygiene contract: importing the module pulls no
``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys


def test_importing_use_unitree_pulls_no_sdk_submodule() -> None:
    importlib.import_module("strands_robots.tools.g1.use_unitree")
    leaked = sorted(m for m in sys.modules if m.startswith("unitree_sdk2py"))
    assert not leaked, f"strands_robots.tools.g1.use_unitree imports pulled SDK submodules: {leaked}"


def test_list_services_names_the_six_sdk_clients() -> None:
    from strands_robots.tools.g1.use_unitree import list_services

    names = {s["service_name"] for s in list_services()}
    assert names == {"loco", "arm", "audio", "motion_switcher", "vui", "robot_state"}


def test_meta_list_services_answers_through_the_tool() -> None:
    from strands_robots.tools.g1.use_unitree import use_unitree

    res = use_unitree("meta", "list_services")
    assert res["status"] == "success"
    assert any(s["service_name"] == "loco" for s in res["result"])


def test_meta_rejects_an_unknown_service() -> None:
    from strands_robots.tools.g1.use_unitree import use_unitree

    res = use_unitree("nope", "Anything")
    assert res["status"] == "error"
    assert "unknown service" in res["message"]


def test_high_danger_set_names_the_collapse_and_walk_ops() -> None:
    from strands_robots.tools.g1.use_unitree import HIGH_DANGER_OPS

    assert ("loco", "ZeroTorque") in HIGH_DANGER_OPS
    assert ("loco", "SetVelocity") in HIGH_DANGER_OPS
    assert ("loco", "Move") in HIGH_DANGER_OPS
    assert ("motion_switcher", "ReleaseMode") in HIGH_DANGER_OPS


def test_mutative_detection_spares_reads_and_flags_writes() -> None:
    from strands_robots.tools.g1.use_unitree import _is_mutative, _is_readonly

    assert _is_readonly("GetFsmId")
    assert _is_readonly("CheckMode")
    assert not _is_mutative("GetVolume")
    assert _is_mutative("SetFsmId")
    assert _is_mutative("ExecuteAction")
    assert _is_mutative("ZeroTorque")
