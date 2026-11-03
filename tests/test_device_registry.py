"""Every registry robot is a device: its manifest is generated from registry data, never written by hand."""

from __future__ import annotations

import json

import pytest

from strands_robots import Description
from strands_robots.description import device_types, registry
from strands_robots.robot.device import ON_DISCONNECT, Manifest, manifest_of

_CANONICAL = sorted({e["name"] for e in registry().values()})
_LOADABLE = [
    n
    for n in _CANONICAL
    if registry()[n].get("asset", {}).get("robot_descriptions_module")
    or (registry()[n].get("asset", {}).get("source") or {}).get("type")
]


def test_registry_device_block_schema():
    for name in _CANONICAL:
        e = registry()[name]
        assert e.get("device", {}).get("type") in device_types(), f"{name}: device.type missing or unknown"
        hw = e.get("hardware") or {}
        assert hw.get("on_disconnect", "hold") in ON_DISCONNECT, name
        for key, bound in (hw.get("safe") or {}).items():
            assert len(bound) == 2 and bound[0] <= bound[1], f"{name}: safe[{key}] {bound}"
        if hw.get("driver") in ("feetech", "microduck"):
            assert hw.get("on_disconnect") == "hold", f"{name}: a gravity-loaded servo arm must hold"


@pytest.mark.network
@pytest.mark.parametrize("name", _LOADABLE)
def test_every_registry_robot_builds_a_manifest(name):
    d = Description.load(name)
    m = manifest_of(d, device_id=name)
    assert Manifest.from_dict(json.loads(json.dumps(m.to_dict()))) == m
    assert m.device_type == registry()[name]["device"]["type"]
    assert [c.name for c in m.controls] == [f"{k}.pos" for k in d.action_keys]
    for c, k in zip(m.controls, d.action_keys, strict=True):
        assert c.range == (tuple(d.limits[k]) if k in d.limits else None)
        if c.range is not None and c.safe is not None:
            assert c.range[0] <= c.safe[0] <= c.safe[1] <= c.range[1]
