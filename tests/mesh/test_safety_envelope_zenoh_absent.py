"""Safety-envelope publish path when the Zenoh stack is not importable.

``Mesh`` publishes e-stop and resume envelopes through four helpers that each
degrade to the transport-agnostic ``put()`` path rather than raising:
:meth:`~strands_robots.mesh.core.Mesh._local_session_zid`,
:meth:`~strands_robots.mesh.core.Mesh._safety_wire_zid`,
:meth:`~strands_robots.mesh.core.Mesh._safety_publisher_for` and
:meth:`~strands_robots.mesh.core.Mesh._publish_safety_envelope`.

Between them they refuse for nineteen distinct reasons -- no session open,
``session.info.zid()`` raising, a ``None`` zid, no publisher declarable,
``declare_publisher`` failing, no ``zenoh.SourceInfo`` constructor, the native
``publisher.put`` raising -- and every one of those is pinned by
``test_safety_envelope_native_path`` / ``test_safety_envelope_fallback`` /
``test_resume_proof_fallback_path``.

The reason pinned here is the one those modules do not reach and the four
docstrings did not enumerate: **the import itself failing**. Two of the four
helpers reach for ``zenoh`` at call time, which is a genuine runtime import on
an install without the ``mesh`` extra; the other two reach for
``strands_robots.mesh.session``, which ``core`` already imports at module scope
(so that arm is defence in depth -- pinned below, along with the property that
makes it unreachable, so it acquires a behavioural test the day that changes).

Both zenoh-absent arms carry a fleet-availability contract on the safety path:

* ``_safety_wire_zid`` must answer ``None`` so an issuer binds the resume
  override proof to the zid-less body the fallback path actually publishes. Its
  own docstring records the alternative: a proof bound to
  ``_local_session_zid`` while the envelope ships stripped "never verifies --
  leaving the fleet stuck in lockout".
* ``_publish_safety_envelope`` must still publish, with ``source_zid`` removed
  from the body. ``_strip_wire_zid``'s docstring records why: a body that still
  advertises ``source_zid`` with no wire ``SourceInfo`` behind it is
  hard-rejected by the receiver, so an unstripped e-stop is a dropped e-stop.
"""

import ast
import inspect
import json
import pathlib
import sys
import types
from unittest.mock import MagicMock

from strands_robots.mesh import core

from .test_resume_proof_fallback_path import _fallback_sample

ESTOP_KEY = "strands/safety/estop"
RESUME_KEY = "strands/safety/resume"


def _hide_zenoh(monkeypatch):
    """Make ``import zenoh`` raise ImportError, as on an install without the
    ``mesh`` extra. A ``None`` entry in ``sys.modules`` is the repository's
    mechanism for an absent optional dependency; ``monkeypatch`` restores it."""
    monkeypatch.setitem(sys.modules, "zenoh", None)


def _zenoh_with_source_info(monkeypatch):
    """Install a zenoh stand-in that DOES carry a ``SourceInfo`` ctor, so the
    native path is reachable. Without this control the absent-arm assertions
    below would also pass on a build that never took the native path at all."""
    fake = types.ModuleType("zenoh")
    fake.SourceInfo = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zenoh", fake)
    return fake


class _RecordingPublisher:
    """A declarable publisher that records any native put it is asked to make."""

    id = "entity-global-id"

    def __init__(self):
        self.puts: list[tuple] = []

    def put(self, payload, **kwargs):
        self.puts.append((payload, kwargs))


class TestTheWireZidDecisionWhenZenohIsAbsent:
    """``_safety_wire_zid`` is the single decision point an issuer consults
    before binding ``source_zid`` into the resume proof."""

    def test_wire_zid_is_none_when_zenoh_is_not_importable(self, monkeypatch):
        m = core.Mesh(robot=object(), peer_id="t1")
        # Both other preconditions satisfied, so the import is what decides.
        monkeypatch.setattr(m, "_local_session_zid", lambda: "deadbeefdeadbeef")
        monkeypatch.setattr(m, "_safety_publisher_for", lambda key: object())
        _hide_zenoh(monkeypatch)

        assert m._safety_wire_zid(RESUME_KEY) is None

    def test_wire_zid_returns_the_zid_when_zenoh_is_importable(self, monkeypatch):
        """Non-vacuity: the same fixture yields the zid once zenoh imports, so
        the refusal above is attributable to the import and nothing else."""
        m = core.Mesh(robot=object(), peer_id="t1")
        monkeypatch.setattr(m, "_local_session_zid", lambda: "deadbeefdeadbeef")
        monkeypatch.setattr(m, "_safety_publisher_for", lambda key: object())
        _zenoh_with_source_info(monkeypatch)

        assert m._safety_wire_zid(RESUME_KEY) == "deadbeefdeadbeef"


class TestTheEnvelopeStillPublishesWhenZenohIsAbsent:
    """The e-stop must reach the wire on a zenoh-free install, in the shape the
    receiver's transport-agnostic accept-without-zid contract requires."""

    def test_estop_publishes_with_the_wire_zid_stripped(self, monkeypatch):
        m = core.Mesh(robot=object(), peer_id="t1")
        publisher = _RecordingPublisher()
        # A publisher IS declarable, so the fallback is reached by the import
        # failing rather than by the earlier no-publisher arm.
        monkeypatch.setattr(m, "_safety_publisher_for", lambda key: publisher)
        _hide_zenoh(monkeypatch)

        published: list[tuple] = []
        monkeypatch.setattr(core, "put", lambda key, payload: published.append((key, payload)))

        m._publish_safety_envelope(
            ESTOP_KEY,
            {"peer_id": "t1", "t": 1.0, "lockout_engaged": True, "source_zid": "deadbeefdeadbeef"},
        )

        assert len(published) == 1, "the e-stop was dropped instead of published"
        key, payload = published[0]
        assert key == ESTOP_KEY
        # Body and wire stay consistently zid-less: an unstripped body with no
        # wire SourceInfo behind it is hard-rejected by the receiver.
        assert "source_zid" not in payload
        assert publisher.puts == [], "no native put is possible without zenoh"
        # Every other field survives the strip.
        assert payload["peer_id"] == "t1"
        assert payload["lockout_engaged"] is True

    def test_a_zid_less_envelope_never_reaches_the_import_at_all(self, monkeypatch):
        """Control: an envelope that requests no attribution takes the earlier
        plain-put arm, so the zenoh-absent behaviour is scoped to the envelopes
        that asked for a wire zid."""
        m = core.Mesh(robot=object(), peer_id="t1")

        def _unreachable(key):  # pragma: no cover - asserted not to run
            raise AssertionError("a zid-less envelope must not resolve a publisher")

        monkeypatch.setattr(m, "_safety_publisher_for", _unreachable)
        _hide_zenoh(monkeypatch)

        published: list[tuple] = []
        monkeypatch.setattr(core, "put", lambda key, payload: published.append((key, payload)))

        m._publish_safety_envelope(ESTOP_KEY, {"peer_id": "t1", "t": 1.0})

        assert published == [(ESTOP_KEY, {"peer_id": "t1", "t": 1.0})]


class TestRemoteResumeStillVerifiesWhenZenohIsAbsent:
    """The end-to-end availability contract: a lockout raised on a zenoh-free
    install must be clearable, or the fleet stays e-stopped forever."""

    def test_resume_proof_verifies_end_to_end(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "operator-secret")

        issuer = core.Mesh(robot=object(), peer_id="issuer")
        issuer.publish_safety_event = MagicMock()
        # A session IS open and a publisher IS declarable -- so binding the
        # proof to _local_session_zid() would be the tempting choice, and the
        # import failure is the only thing that rules the native path out.
        monkeypatch.setattr(issuer, "_local_session_zid", lambda: "deadbeefdeadbeef")
        monkeypatch.setattr(issuer, "_safety_publisher_for", lambda key: _RecordingPublisher())
        _hide_zenoh(monkeypatch)

        published: dict = {}

        def capture_put(key, payload):
            published["key"] = key
            published["payload"] = payload

        monkeypatch.setattr(core, "put", capture_put)

        issuer._estop_lockout.set()
        issuer._last_estop_ts = core.time.time()
        assert issuer._resume_lockout("operator-secret") == {"status": "ok"}

        assert published["key"] == RESUME_KEY
        envelope = published["payload"]
        assert "source_zid" not in envelope, "the fallback path publishes a zid-less body"
        assert "override_proof" in envelope

        # A receiver on the same zenoh-free transport accepts the proof and
        # clears its lockout: issuer and receiver agree on the MAC input.
        receiver = core.Mesh(robot=object(), peer_id="receiver")
        receiver.publish_safety_event = MagicMock()
        receiver._estop_lockout.set()
        assert receiver._estop_lockout.is_set()

        receiver._on_safety_resume(_fallback_sample(envelope))

        assert receiver._estop_lockout.is_set() is False, "the fleet would stay e-stopped"

    def test_the_proof_is_bound_to_the_published_body(self, monkeypatch):
        """The MAC input must be the exact bytes the receiver recomputes over:
        a proof bound to a zid the body does not carry never verifies."""
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "operator-secret")

        issuer = core.Mesh(robot=object(), peer_id="issuer")
        issuer.publish_safety_event = MagicMock()
        monkeypatch.setattr(issuer, "_local_session_zid", lambda: "deadbeefdeadbeef")
        monkeypatch.setattr(issuer, "_safety_publisher_for", lambda key: _RecordingPublisher())
        _hide_zenoh(monkeypatch)

        published: dict = {}
        monkeypatch.setattr(core, "put", lambda key, payload: published.update(payload=payload))

        issuer._estop_lockout.set()
        issuer._last_estop_ts = core.time.time()
        issuer._resume_lockout("operator-secret")

        envelope = published["payload"]
        expected = core.hmac.new(
            b"operator-secret",
            json.dumps(
                {
                    "peer_id": "issuer",
                    "t": envelope["t"],
                    "lockout_elapsed_s": envelope["lockout_elapsed_s"],
                    "proof_nonce": envelope["proof_nonce"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            "sha256",
        ).hexdigest()
        assert envelope["override_proof"] == expected


class TestTheSessionModuleArmsAreDefenceInDepth:
    """``_local_session_zid`` / ``_safety_publisher_for`` also tolerate their
    in-package import failing. That arm cannot fire in a complete install --
    pinned here, together with the property that makes it unreachable."""

    def test_the_session_module_import_cannot_fail_in_a_complete_install(self):
        module_level = {
            node.module
            for node in ast.parse(inspect.getsource(core)).body
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "strands_robots.mesh.session" in module_level, (
            "core no longer imports the session module at module scope, so the "
            "function-local import in _local_session_zid / _safety_publisher_for "
            "can now fail at runtime -- the arms below stop being defence in depth"
        )
        assert "strands_robots.mesh.session" in sys.modules

    def test_local_session_zid_refuses_when_the_session_module_is_hidden(self, monkeypatch):
        m = core.Mesh(robot=object(), peer_id="t1")
        monkeypatch.setitem(sys.modules, "strands_robots.mesh.session", None)

        assert m._local_session_zid() is None

    def test_publisher_for_refuses_when_the_session_module_is_hidden(self, monkeypatch):
        m = core.Mesh(robot=object(), peer_id="t1")
        monkeypatch.setitem(sys.modules, "strands_robots.mesh.session", None)

        assert m._safety_publisher_for(ESTOP_KEY) is None


def test_every_documented_refusal_reason_names_the_import():
    """The four docstrings enumerate the reasons each helper degrades. The
    import failing was absent from all four while no test reached it; keep the
    enumeration and the behaviour in step."""
    for name in ("_local_session_zid", "_safety_wire_zid", "_safety_publisher_for", "_publish_safety_envelope"):
        doc = inspect.getdoc(getattr(core.Mesh, name)) or ""
        assert "import" in doc.lower(), f"{name} does not enumerate the import-failure reason"


def test_the_module_under_test_is_the_installed_one():
    """Non-vacuity: assert the helpers come from the package source tree rather
    than a stale copy on sys.path."""
    assert pathlib.Path(inspect.getfile(core)).name == "core.py"
    assert pathlib.Path(inspect.getfile(core)).parent.name == "mesh"
