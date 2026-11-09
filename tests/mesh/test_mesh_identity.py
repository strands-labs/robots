"""Mesh identities: a CA, certificates that name a peer and the devices it serves, signed envelopes."""

import os
import stat
import time

import numpy as np
import pytest

pytest.importorskip("cryptography")

from strands_robots.mesh import pki, wire  # noqa: E402
from strands_robots.mesh.identity import Freshness, Identity, Unverified  # noqa: E402


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab")
    ca = pki.make_ca(root / "ca")
    ids = {
        "device": pki.issue(ca, "lab-pc", root / "device", serves=["so101", "arm-*"]),
        "operator": pki.issue(ca, "alice", root / "operator"),
        "insider": pki.issue(ca, "bob", root / "insider"),
    }
    other = pki.make_ca(root / "other-ca")
    ids["rogue"] = pki.issue(other, "alice", root / "rogue", serves=["*"])
    return root, {k: Identity.load(v) for k, v in ids.items()}


def _env(peer="alice.1", **kw):
    return wire.envelope("lab", "so101", "rpc", "act", seq=1, args={"action": np.arange(3.0)}, peer=peer, **kw)


def test_keys_are_private_and_a_ca_is_never_overwritten(lab):
    root, _ = lab
    for path in (root / "ca" / "ca.key", root / "device" / "key.pem"):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    with pytest.raises(FileExistsError):
        pki.make_ca(root / "ca")


def test_a_certificate_names_its_peer_and_the_devices_it_serves(lab):
    _, ids = lab
    dev, op = ids["device"].signer, ids["operator"].signer
    assert (dev.name, dev.serves) == ("lab-pc", ("so101", "arm-*"))
    assert dev.may_serve("so101") and dev.may_serve("arm-7") and not dev.may_serve("so100")
    assert op.serves == () and not op.may_serve("so101")
    assert op.owns("alice.abc") and not op.owns("alice") and not op.owns("alicex.abc") and not op.owns("bob.abc")


@pytest.mark.parametrize("bad", ["a.b", "", "x" * 65, "a b", "a/b"])
def test_a_peer_name_has_no_dots_or_separators(bad, tmp_path):
    with pytest.raises(ValueError, match="no dots"):
        pki.check_name(bad)


def test_a_sealed_envelope_survives_the_wire_and_names_its_signer(lab):
    _, ids = lab
    raw = wire.encode(ids["operator"].seal(_env()))
    env = wire.decode_envelope(raw)
    assert ids["device"].verify(env).name == "alice"


@pytest.mark.parametrize(
    "tamper",
    [
        lambda e: e.update(peer="bob.1"),
        lambda e: e.update(seq=2),
        lambda e: e["args"].update(action=np.arange(4.0)),
        lambda e: e.update(sig={**e["sig"], "sig": e["sig"]["sig"][:-4] + "AAAA"}),
        lambda e: e.pop("sig"),
        lambda e: e.update(sig="x"),
    ],
)
def test_a_changed_or_unsigned_envelope_is_refused(lab, tamper):
    _, ids = lab
    env = wire.decode_envelope(wire.encode(ids["operator"].seal(_env())))
    tamper(env)
    with pytest.raises(Unverified):
        ids["device"].verify(env)


def test_a_certificate_from_another_ca_is_refused(lab):
    _, ids = lab
    env = wire.decode_envelope(wire.encode(ids["rogue"].seal(_env())))
    with pytest.raises(Unverified, match="not issued by this mesh's CA"):
        ids["device"].verify(env)


def test_one_signer_cannot_carry_another_signers_certificate(lab):
    _, ids = lab
    stolen = ids["operator"].seal(_env())
    env = ids["insider"].seal(_env(peer="alice.1"))
    env["sig"]["cert"] = stolen["sig"]["cert"]
    with pytest.raises(Unverified, match="does not hold"):
        ids["device"].verify(wire.decode_envelope(wire.encode(env)))


def test_a_repeated_or_stale_request_is_refused():
    fresh = Freshness(max_skew=5.0)
    now = time.time()
    fresh.admit("alice.1", 1, now)
    fresh.admit("alice.1", 2, now)
    fresh.admit("bob.1", 1, now)
    with pytest.raises(Unverified, match="already answered"):
        fresh.admit("alice.1", 1, now)
    with pytest.raises(Unverified, match="off this device's clock"):
        fresh.admit("alice.1", 3, now - 6.0)
