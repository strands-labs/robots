"""Regression pin: _publish_safety_envelope fallback must strip body
source_zid so receivers do not hard-reject body-present + wire-absent
envelopes (availability fix)."""


def test_estop_fallback_strips_source_zid(monkeypatch):
    from strands_robots.mesh import core

    captured = {}

    def fake_put(key, payload):
        captured["key"] = key
        captured["payload"] = payload

    monkeypatch.setattr(core, "put", fake_put)

    m = core.Mesh(robot=object(), peer_id="t1")
    # Force the fallback path (no publisher available).
    monkeypatch.setattr(m, "_safety_publisher_for", lambda key: None)

    m._publish_safety_envelope(
        "strands/safety/estop",
        {"peer_id": "t1", "t": 1.0, "source_zid": "deadbeef"},
    )

    assert captured["key"] == "strands/safety/estop"
    assert "source_zid" not in captured["payload"]
    # Other fields preserved.
    assert captured["payload"]["peer_id"] == "t1"


def test_strip_wire_zid_noop_when_absent():
    from strands_robots.mesh import core

    m = core.Mesh(robot=object(), peer_id="t2")
    payload = {"peer_id": "t2", "t": 1.0}
    # No source_zid -> returns same object (cheap no-op).
    assert m._strip_wire_zid(payload) is payload
