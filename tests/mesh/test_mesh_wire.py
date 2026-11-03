"""The mesh envelope and key space: one versioned shape, numpy-safe, one table of keys."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.mesh import wire


def test_envelope_round_trips_with_numpy_payloads():
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    env = wire.envelope("lab-a", "so101-a", "rpc", "act", seq=7, args={"q": np.array([0.1, 0.2]), "img": img})
    back = wire.decode(wire.encode(env))
    assert back["v"] == wire.VERSION == 1
    assert (back["tenant"], back["device_id"], back["seq"], back["kind"], back["name"]) == (
        "lab-a",
        "so101-a",
        7,
        "rpc",
        "act",
    )
    assert isinstance(back["t"], float)
    np.testing.assert_array_equal(back["args"]["q"], [0.1, 0.2])
    assert back["args"]["img"].dtype == np.uint8 and back["args"]["img"].shape == (2, 2, 3)
    np.testing.assert_array_equal(back["args"]["img"], img)


def test_numpy_scalars_encode_as_plain_numbers():
    back = wire.decode(wire.encode({"a": np.float32(1.5), "b": np.int64(3), "c": np.bool_(True)}))
    assert back == {"a": 1.5, "b": 3, "c": True}


def test_unknown_version_is_refused_loudly():
    raw = wire.encode({**wire.envelope("t", "d", "rpc", "act"), "v": 99})
    with pytest.raises(ValueError, match="version 99"):
        wire.decode_envelope(raw)


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="kind"):
        wire.envelope("t", "d", "bogus", "x")


def test_key_space_is_one_table():
    assert wire.key("manifest", "lab-a", "so101-a") == "strands/lab-a/so101-a/manifest"
    assert wire.key("state", "lab-a", "so101-a") == "strands/lab-a/so101-a/state"
    assert wire.key("event", "lab-a", "so101-a", "estop") == "strands/lab-a/so101-a/event/estop"
    assert wire.key("rpc", "lab-a", "so101-a", "act") == "strands/lab-a/so101-a/rpc/act"
    assert wire.key("presence", "lab-a", "so101-a") == "strands/lab-a/@presence/so101-a"
    assert wire.key("presence", "lab-a", "*") == "strands/lab-a/@presence/*"
    assert wire.parse_presence("strands/lab-a/@presence/so101-a") == ("lab-a", "so101-a")


@pytest.mark.parametrize("bad", ["", "a/b", "a*b", "a$b", "@x", "x?y", "a#b"])
def test_ids_that_would_break_the_key_space_are_refused(bad):
    with pytest.raises(ValueError):
        wire.key("state", "lab-a", bad)


def test_object_arrays_never_leave_the_process():
    with pytest.raises(TypeError, match="object arrays"):
        wire.encode({"x": np.array([object(), 1], dtype=object)})
    forged = b'{"x":{"__nd__":["|O",[2],"ACruBAEAAAD4gUYHAQAAAA=="]}}'
    with pytest.raises(ValueError, match="object arrays"):
        wire.decode(forged)


def test_nan_and_inf_are_refused_not_sent_as_invalid_json():
    with pytest.raises(ValueError, match="NaN"):
        wire.encode({"x": float("nan")})
    with pytest.raises(ValueError, match="NaN"):
        wire.encode({"x": np.array([1.0, np.inf])})


@pytest.mark.parametrize(
    "patch",
    [{"kind": "bogus"}, {"tenant": "a/b"}, {"device_id": ""}, {"name": 3}, {"seq": "x"}, {"t": "now"}],
)
def test_decode_envelope_validates_every_field_a_peer_controls(patch):
    raw = wire.encode({**wire.envelope("t", "d", "rpc", "act"), **patch})
    with pytest.raises(ValueError):
        wire.decode_envelope(raw)


def test_a_bare_version_is_not_an_envelope():
    with pytest.raises(ValueError):
        wire.decode_envelope(wire.encode({"v": 1}))


def test_envelope_ids_must_be_keyable():
    with pytest.raises(ValueError):
        wire.envelope("a/b", "d", "rpc", "act")


def test_key_name_is_required_exactly_where_the_slot_has_one():
    with pytest.raises(ValueError, match="needs a name"):
        wire.key("rpc", "t", "d")
    with pytest.raises(ValueError, match="takes no name"):
        wire.key("presence", "t", "d", "x")


def test_presence_is_outside_the_tenant_wildcard():
    """``@`` segments only match verbatim: ``strands/<tenant>/**`` never sees presence keys."""
    from strands_robots.mesh.transports import key_matches

    k = wire.key("presence", "lab", "d1")
    assert key_matches(wire.key("presence", "lab", "*"), k)
    assert not key_matches("strands/lab/**", k)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nan_literals_from_a_peer_are_refused(literal):
    with pytest.raises(ValueError, match="refusing"):
        wire.decode(f'{{"x": {literal}}}')
