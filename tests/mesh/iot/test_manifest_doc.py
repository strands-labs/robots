"""The ``manifest`` shadow fits every registry robot under IoT's 8 KB cap, and says where the full record is."""

from __future__ import annotations

import json

import pytest

from strands_robots.description import Description
from strands_robots.description.registry import registry, urdf_registry
from strands_robots.mesh.iot.manifest import manifest_doc
from strands_robots.mesh.iot.shadow import SHADOW_DOC_CAP
from strands_robots.mesh.record import record
from strands_robots.mesh.serve import ARGS

_CANONICAL = sorted({e["name"] for e in registry().values()})
_NO_MODEL = {
    n
    for n in _CANONICAL
    if not registry()[n].get("asset", {}).get("robot_descriptions_module")
    and not (registry()[n].get("asset", {}).get("source") or {}).get("type")
    and n not in urdf_registry()
}
_BIG = ("unitree_h1_2", "unitree_g1", "aloha", "shadow_hand", "rby1", "so101")  # 51, 29, 14, 20, 24, 6 controls


def _doc_bytes(name: str) -> tuple[int, dict]:
    pytest.importorskip("mujoco")
    from strands_robots import Robot, Simulation

    sim = Simulation()
    try:
        rec = record(Robot(name, sim=sim), name, {"role": "arm", "site": "lab-1"}, "lab", tuple(ARGS))
    finally:
        sim.close()
    doc = manifest_doc(rec)
    return len(json.dumps({"state": {"reported": doc}}, separators=(",", ":")).encode()), doc


@pytest.mark.parametrize("name", _BIG)
def test_the_manifest_shadow_fits_the_biggest_registry_robots_with_room(name):
    size, doc = _doc_bytes(name)
    assert size <= SHADOW_DOC_CAP // 2, f"{name}: {size} B (cap {SHADOW_DOC_CAP})"
    assert doc["dof"] == len(doc["controls"]) and doc["full"] == f"strands/lab/{name}/manifest"


@pytest.mark.network
@pytest.mark.parametrize("name", [n for n in _CANONICAL if n not in _NO_MODEL])
def test_the_manifest_shadow_fits_every_registry_robot(name):
    d = Description.load(name)
    rec = {
        "id": name,
        "device_type": name,
        "tenant": "lab",
        "labels": {},
        "manifest": {
            "controls": [
                {"name": f"{k}.pos", "unit": "rad", "range": list(d.limits.get(k, (-3.14, 3.14)))}
                for k in d.action_keys
            ],
            "sensors": [
                {"name": f"{k}.{s}", "unit": "rad", "kind": "joint"} for k in d.joints for s in ("pos", "vel", "effort")
            ],
            "rpcs": [{"name": r} for r in ARGS],
            "events": [{"name": e} for e in ("state", "fault", "estop", "run.started", "run.stopped")],
        },
    }
    size = len(json.dumps({"state": {"reported": manifest_doc(rec)}}, separators=(",", ":")).encode())
    assert size <= SHADOW_DOC_CAP, f"{name}: {size} B for {d.dof} controls"


def test_the_manifest_shadow_keeps_identity_ranges_units_and_names_only():
    rec = {
        "id": "arm-1",
        "device_type": "robot_arm",
        "tenant": "lab",
        "labels": {"role": "arm"},
        "manifest": {
            "device_id": "arm-1",
            "name": "so101",
            "real": True,
            "twin_of": None,
            "controls": [
                {
                    "name": "shoulder_pan.pos",
                    "unit": "rad",
                    "range": [-3.0, 3.0],
                    "safe": [-1.5, 1.5],
                    "interlocks": [],
                },
                {"name": "gripper.pos", "unit": "range_0_100", "range": [0, 100], "on_disconnect": "hold"},
            ],
            "sensors": [{"name": "shoulder_pan.pos", "kind": "joint"}, {"name": "front", "kind": "camera"}],
            "rpcs": [{"name": "act", "params": {"type": "object"}, "doc": "…"}, {"name": "estop", "params": {}}],
            "events": [{"name": "state", "hz": 10.0}, {"name": "estop", "hz": None}],
            "description": {"dof": 6, "action_keys": ["…"] * 6},
        },
    }
    doc = manifest_doc(rec)
    assert doc["controls"] == {"shoulder_pan.pos": [-1.5, 1.5], "gripper.pos": [0.0, 100.0]}  # safe wins over range
    assert doc["units"] == {"rad": ["shoulder_pan.pos"], "range_0_100": ["gripper.pos"]}
    assert doc["sensors"] == {"joint": 1, "camera": 1} and doc["rpcs"] == ["act", "estop"]
    assert doc["events"] == ["state", "estop"] and doc["dof"] == 2 and doc["real"] is True
    assert doc["full"] == "strands/lab/arm-1/manifest" and "description" not in doc
    assert doc["id"] == doc["device_id"] == "arm-1" and doc["labels"] == {"role": "arm"}
