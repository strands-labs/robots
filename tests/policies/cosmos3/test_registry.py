"""Cosmos 3 registry + factory resolution tests."""

from strands_robots.policies import create_policy, list_providers
from strands_robots.policies.cosmos3 import Cosmos3Policy
from strands_robots.policies.cosmos3.client import Cosmos3WebsocketClient
from strands_robots.registry import get_policy_provider, list_policy_providers, resolve_policy


def test_cosmos3_in_registry():
    assert "cosmos3" in list_policy_providers()
    cfg = get_policy_provider("cosmos3")
    assert cfg["module"] == "strands_robots.policies.cosmos3"
    assert cfg["class"] == "Cosmos3Policy"


def test_shorthands_resolve_to_cosmos3():
    """The canonical short name is ``cosmos3`` (with ``c3`` alias). The
    bare ``cosmos`` shorthand was intentionally dropped — it's ambiguous
    against future NVIDIA Cosmos products (Predict, Reason, Transfer, etc.)
    and was a one-way-door if we'd kept it. Pinning the new contract here
    so a future copy-paste cannot silently re-introduce the collision."""
    for name in ("cosmos3", "c3"):
        prov, _ = resolve_policy(name)
        assert prov == "cosmos3", name


def test_bare_cosmos_shorthand_does_not_resolve_to_cosmos3():
    """Bare ``cosmos`` must NOT resolve to the cosmos3 provider — it falls
    back to lerobot_local (the unrecognized-policy default) so a future
    NVIDIA Cosmos provider can claim that namespace cleanly."""
    prov, _ = resolve_policy("cosmos")
    assert prov != "cosmos3"


def test_cosmos3_url_pattern():
    # Pin host/port round-trip so a future regression cannot silently drop
    # the URL kwargs and quietly route every cosmos3:// call to the default
    # localhost:8000 (the failure mode reported on PR #317 R3).
    prov, kwargs = resolve_policy("cosmos3://localhost:8000")
    assert prov == "cosmos3"
    assert kwargs["host"] == "localhost"
    assert kwargs["port"] == 8000

    prov, kwargs = resolve_policy("cosmos3://prod-server:9000")
    assert prov == "cosmos3"
    assert kwargs["host"] == "prod-server"
    assert kwargs["port"] == 9000

    # Bare host (no port) defaults to 8000, mirroring the wss:// branch.
    prov, kwargs = resolve_policy("cosmos3://otherhost")
    assert prov == "cosmos3"
    assert kwargs["host"] == "otherhost"
    assert kwargs["port"] == 8000


def test_model_id_override_disambiguates_from_groot():
    # nvidia/Cosmos3-* must route to cosmos3, not groot
    prov, kwargs = resolve_policy("nvidia/Cosmos3-Nano-Policy-DROID")
    assert prov == "cosmos3"


def test_listed_in_providers():
    assert "cosmos3" in list_providers()


def test_create_policy_constructs_cosmos3():
    # No server needed: client connects lazily, construction must not touch network.
    p = create_policy("cosmos3", embodiment="droid", port=8123)
    assert isinstance(p, Cosmos3Policy)
    assert p.provider_name == "cosmos3"
    assert p.port == 8123
    assert isinstance(p._client, Cosmos3WebsocketClient)
