"""A discovered device and the selector grammar ``device(k:v,…)`` / ``device(*)``."""

from __future__ import annotations

import pytest

from strands_robots.mesh.device import Device, Selector


def _dev(**labels):
    return Device(id="so101-a", device_type="so101", tenant="lab-a", labels=labels, manifest={}, last_seen=1.0)


def test_star_matches_everything():
    assert Selector.parse("device(*)").matches(_dev())


def test_labels_and_reserved_keys_all_must_match():
    d = _dev(role="arm", bench="a")
    assert Selector.parse("device(role:arm)").matches(d)
    assert Selector.parse("device(role:arm, bench:a)").matches(d)
    assert not Selector.parse("device(role:arm,bench:desk)").matches(d)
    assert Selector.parse("device(type:so101)").matches(d)
    assert Selector.parse("device(id:so101-a)").matches(d)
    assert Selector.parse("device(tenant:lab-a)").matches(d)
    assert not Selector.parse("device(tenant:lab-b)").matches(d)
    assert not Selector.parse("device(color:red)").matches(d)


def test_a_bare_id_is_sugar_for_id_selector():
    assert Selector.parse("so101-a") == Selector.parse("device(id:so101-a)")


@pytest.mark.parametrize("bad", ["device(", "device(role)", "device(:x)", "device(role:)", "dev(role:arm)", ""])
def test_malformed_selectors_are_refused_with_the_grammar(bad):
    with pytest.raises(ValueError, match=r"device\(<key>:<value>"):
        Selector.parse(bad)


def test_device_round_trips_as_a_dict():
    d = _dev(role="arm")
    assert Device.from_dict(d.to_dict()) == d
