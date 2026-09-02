"""Q10 + Q11: the relying-party id came from the caller, and the challenge stash
had no bound.

Q10: ``_derive_rp_id`` returned the client's ``Host`` header, and
``finish_registration`` verified against that same claimed value -- self-consistent
and enforcing nothing. A credential is cryptographically bound to its rp_id, so the
store is the only authority worth consulting.

Q11: ``_stash_challenge`` pruned only by TTL, only on insert. The measured growth
was slow (~0.5KB each), so the real problem is not memory: it is that one
unauthenticated caller could hold an unbounded share of a public table and push out
the operator's pending login. A global cap alone does not fix that -- per-client
fairness does.
"""

from __future__ import annotations

import types

import pytest

import strands_robots.dashboard.auth as auth
from strands_robots.dashboard.auth import known_rp_ids, rp_id_verdict


def _req(host: str = "robots.cagatay.my", **headers):
    h = {"host": host, **headers}
    return types.SimpleNamespace(
        headers=types.SimpleNamespace(get=lambda k, d=None: h.get(k, d)),
        client=types.SimpleNamespace(host="203.0.113.9"),
    )


# --------------------------------------------------------------------------
# Q10 - who decides the rp_id
# --------------------------------------------------------------------------


def test_explicit_config_beats_a_claimed_host():
    assert rp_id_verdict("evil.example", "robots.cagatay.my")[0] == "robots.cagatay.my"


def test_loopback_outranks_even_the_pin():
    """A browser at http://localhost cannot use a pinned domain as its rp_id --
    the spec requires a registrable suffix of the origin, so honouring the pin
    would make the browser refuse the ceremony and lock out the local door."""
    rp, why = rp_id_verdict("localhost", "robots.cagatay.my")
    assert rp == "localhost" and why == "loopback"


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_loopback_is_always_allowed(host):
    """A browser on this machine is the operator; local dev must not depend on
    what a remote store happens to record."""
    assert rp_id_verdict(host, "", {"robots.cagatay.my"})[0] == host
    assert rp_id_verdict(host, "robots.cagatay.my")[0] == host


def test_a_host_matching_an_enrolled_credential_is_allowed():
    rp, why = rp_id_verdict("robots.cagatay.my", "", {"robots.cagatay.my"})
    assert rp == "robots.cagatay.my" and "enrolled" in why


def test_a_spoofed_host_is_refused_once_a_binding_is_known():
    rp, why = rp_id_verdict("evil.example", "", {"robots.cagatay.my"})
    assert rp is None
    assert "evil.example" in why and "robots.cagatay.my" in why


def test_a_store_with_no_recorded_binding_still_allows_first_enrollment():
    """Fresh install (and stores written before rp_ids were recorded): something
    has to be bindable, and that path is bootstrap-token gated."""
    rp, why = rp_id_verdict("robots.cagatay.my", "", set())
    assert rp == "robots.cagatay.my" and "legacy" in why


def test_derive_rp_id_raises_a_400_for_a_refused_host(monkeypatch):
    monkeypatch.setattr(auth, "_forced_rp_id", lambda: "")
    monkeypatch.setattr(auth, "known_rp_ids", lambda store=None: {"robots.cagatay.my"})
    with pytest.raises(auth.HTTPException) as e:
        auth._derive_rp_id(_req("evil.example"))
    assert e.value.status_code == 400
    assert "not one of the enrolled" in str(e.value.detail)


def test_derive_rp_id_returns_the_host_when_it_is_allowed(monkeypatch):
    monkeypatch.setattr(auth, "_forced_rp_id", lambda: "")
    monkeypatch.setattr(auth, "known_rp_ids", lambda store=None: {"robots.cagatay.my"})
    assert auth._derive_rp_id(_req("robots.cagatay.my")) == "robots.cagatay.my"


def test_known_rp_ids_ignores_credentials_that_never_recorded_one():
    store = {
        "credentials": [
            {"id": "a", "name": "legacy"},  # enrolled before this
            {"id": "b", "rp_id": "robots.cagatay.my"},
            {"id": "c", "rp_id": ""},  # empty is not a binding
        ]
    }
    assert known_rp_ids(store) == {"robots.cagatay.my"}


def test_a_legacy_only_store_contributes_nothing_and_stays_usable():
    store = {"credentials": [{"id": "a", "name": "cagatay"}]}
    assert known_rp_ids(store) == set()
    # ...so the owner's existing passkey keeps working through the tunnel.
    assert rp_id_verdict("robots.cagatay.my", "", known_rp_ids(store))[0] == "robots.cagatay.my"


# --------------------------------------------------------------------------
# Q11 - the challenge table has a bound, and one client cannot own it
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_stash():
    auth._challenges.clear()
    yield
    auth._challenges.clear()


def test_one_client_cannot_hold_more_than_its_share(monkeypatch):
    monkeypatch.setattr(auth, "_CHAL_MAX_PER_IP", 8)
    for _ in range(200):
        auth._stash_challenge("auth", b"x", {}, ip="198.51.100.7")
    assert len(auth._challenges) == 8


def test_a_flood_cannot_evict_another_client_s_pending_login(monkeypatch):
    """The property a global cap alone does NOT give."""
    monkeypatch.setattr(auth, "_CHAL_MAX_PER_IP", 4)
    monkeypatch.setattr(auth, "_CHAL_MAX", 32)
    mine = auth._stash_challenge("auth", b"operator", {}, ip="192.0.2.5")
    for _ in range(500):
        auth._stash_challenge("reg", b"flood", {}, ip="198.51.100.7")
    assert mine in auth._challenges  # the operator's ceremony survived


def test_the_table_has_a_global_bound_even_across_many_clients(monkeypatch):
    monkeypatch.setattr(auth, "_CHAL_MAX", 40)
    monkeypatch.setattr(auth, "_CHAL_MAX_PER_IP", 4)
    for i in range(400):
        auth._stash_challenge("auth", b"x", {}, ip=f"10.0.{i // 256}.{i % 256}")
    assert len(auth._challenges) <= 40


def test_an_unattributable_caller_still_gets_a_challenge(monkeypatch):
    """No ip (odd transport) must not mean no login."""
    cid = auth._stash_challenge("auth", b"x", {}, ip=None)
    assert cid in auth._challenges


def test_a_stashed_challenge_still_pops_normally():
    cid = auth._stash_challenge("auth", b"secret", {"rp_id": "localhost"}, ip="192.0.2.5")
    rec = auth._pop_challenge(cid, "auth")
    assert rec["challenge"] == b"secret" and rec["extra"]["rp_id"] == "localhost"


def test_the_client_ip_prefers_a_forwarded_header_for_fairness_only():
    """Attacker-settable, so it is used to SPREAD the cap, never to grant trust."""
    assert auth._client_ip(_req(**{"cf-connecting-ip": "198.51.100.7"})) == "198.51.100.7"
    assert auth._client_ip(_req(**{"x-forwarded-for": "198.51.100.7, 10.0.0.1"})) == "198.51.100.7"
    assert auth._client_ip(_req()) == "203.0.113.9"  # falls back to the socket


def test_client_ip_never_raises_on_a_strange_object():
    assert auth._client_ip(object()) is None
