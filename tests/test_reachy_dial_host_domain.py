"""``ReachyMiniDriver``'s ``host`` is the shared dialled-host domain.

The port half of the address this driver dials is held to a shared domain, for
the reason :mod:`tests.test_reachy_api_port_domain` records: nothing downstream
refuses it, so an unusable port is reported as an unreachable daemon. ``host`` is
interpolated into the same expression - ``http://<host>:<api_port>`` in
:func:`strands_robots.drivers.reachy_transport.api`, and
``ws://<host>:<api_port>/ws/sdk`` in ``WebSocketLink`` - and was held to nothing.

The host half is the stronger case, because it can discard the guarded port:

* A URI delimiter re-cuts the URL and the validated port is the component it
  takes. ``host="127.0.0.1/foo"`` builds ``http://127.0.0.1/foo:8000/...``, which
  resolves as host ``127.0.0.1`` with the port in the path, so the driver dials
  :80 - a port nobody configured. The port domain cannot see this, because it is
  the host half that discards the port.
* Userinfo redirects the dial outright: ``"bot.local@evil.example"`` resolves to
  ``evil.example`` on the configured port.
* A non-string is carried verbatim, so ``None`` reaches the resolver as the DNS
  name ``"none"`` and is reported in the device identity as
  ``"Reachy Mini @ None"``.

``TestWhyTheConstructorOwnsTheDomain`` pins those premises rather than asserting
them in prose; they hold on either tree, and they are what makes the constructor
the right owner. What this surface shares with the policy clients that dial the
same shape - that each unusable spelling is refused, that the refusal names the
host, and that a caller who gets both halves wrong is told about the host and not
the port it would have discarded - is pinned for all of them together in
:mod:`tests.test_dial_host_is_graded_wherever_the_port_is` rather than restated
here. The documented fail-safe that reads an unreachable daemon as the
Wireless variant is deliberately unchanged - a daemon that is down is not a
caller mistake.
"""

from __future__ import annotations

import ast
import asyncio
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

import strands_robots
from strands_robots.utils import dial_host_error
from tests._device_connect_real import use_the_real_edge
from tests._package_ast import parse_source, walk_tree
from tests.test_reachy_api_port_domain import _exported_names

# Values that cannot address the host half of the daemon URL, with what each one
# does when interpolated into it anyway.
UNUSABLE_HOSTS: list[Any] = [
    "127.0.0.1/foo",  # path takes the validated port; dials :80
    "bot.local?x=1",  # query delimiter, same discard
    "http://bot.local",  # a pasted URI resolves to host "http"
    "bot.local@evil.example",  # userinfo: resolves to evil.example
    "bot.local:9999",  # a second port in the host half
    "",  # names no host at all
    "bot\tlocal",  # tab silently dropped by the resolver
    "bot.local\x00",  # NUL truncates the lookup
    None,  # carried verbatim as the DNS name "none"
    8000,
    ["bot.local"],
]

# Every spelling a Reachy Mini is actually reached by must keep working.
USABLE_HOSTS = ["reachy-mini.local", "192.168.1.42", "localhost", "0.0.0.0", "[::1]"]

# The values whose delimiter discards the guarded port, and the host each one
# leaves the client dialling instead.
PORT_DISCARDING_HOSTS = [("127.0.0.1/foo", "127.0.0.1"), ("bot.local?x=1", "bot.local"), ("http://bot.local", "http")]


@pytest.fixture
def rmd():
    """The reachy_mini_driver module bound to the real device_connect_edge."""
    use_the_real_edge()
    import strands_robots.device_connect.reachy_mini_driver as module

    return module


class TestDialHostDomain:
    """The constructor accepts exactly the hosts the shared domain accepts."""

    def test_a_usable_host_is_accepted_and_stored(self, rmd):
        """Every host a Mini is reached by still constructs, carried verbatim."""
        for host in USABLE_HOSTS:
            assert rmd.ReachyMiniDriver(host=host)._host == host, f"{host!r}"

    def test_the_default_host_is_usable(self, rmd):
        """The documented default must satisfy the domain it now enforces."""
        assert rmd.ReachyMiniDriver()._host == "reachy-mini.local"

    def test_the_accepted_domain_is_the_shared_dial_host_domain(self, rmd):
        """The driver refuses a host iff the shared domain refuses it.

        Asserted as an equivalence so the two cannot drift: the same value must
        not be refused by one surface that dials a service and accepted by the
        next.
        """
        for host in UNUSABLE_HOSTS + USABLE_HOSTS:
            shared_refuses = dial_host_error(host, "host", "ReachyMiniDriver") is not None
            try:
                rmd.ReachyMiniDriver(host=host)
                driver_refuses = False
            except ValueError:
                driver_refuses = True
            assert driver_refuses is shared_refuses, f"{host!r}"


class TestWhyTheConstructorOwnsTheDomain:
    """Premises for the guard's placement. These hold on either tree."""

    @pytest.mark.parametrize(("host", "resolved"), PORT_DISCARDING_HOSTS)
    def test_a_delimiter_in_the_host_discards_the_validated_port(self, host, resolved):
        """The guarded port lands in the path, so the client dials :80.

        This is why the port domain beside it cannot cover this: the port it
        validated is not the port that gets dialled.
        """
        url = urlsplit(f"http://{host}:8000/api/daemon/status")
        assert url.hostname == resolved
        assert url.port is None

    def test_userinfo_in_the_host_redirects_the_dial(self):
        """The authority is what follows ``@``, so another host is reached."""
        assert urlsplit("http://bot.local@evil.example:8000/api/daemon/status").hostname == "evil.example"

    def test_the_daemon_url_interpolates_the_host_verbatim(self, monkeypatch):
        """``api`` builds ``http://<host>:port/path`` with no coercion."""
        from strands_robots.drivers import reachy_transport

        captured: list[str] = []

        def spy(req, body=None, timeout=None, **kwargs):
            captured.append(req.full_url)
            raise urllib.error.URLError("test: never dialed")

        monkeypatch.setattr(urllib.request, "urlopen", spy)
        for host in ["127.0.0.1/foo", None, 8000]:
            captured.clear()
            reachy_transport.api(host, 8000, "/api/daemon/status")
            assert captured == [f"http://{host}:8000/api/daemon/status"], f"{host!r}"

    def test_the_websocket_target_carries_the_same_value(self):
        """The Lite link interpolates the host into its own ``ws://`` target."""
        from strands_robots.drivers import reachy_transport

        assert reachy_transport.WebSocketLink("127.0.0.1/foo", 8000)._host == "127.0.0.1/foo"


class TestTheRefusalPrecedesAnyState:
    """A refused host allocates nothing and reaches nothing."""

    def test_a_refused_host_allocates_no_base_driver_state(self, rmd, monkeypatch):
        """The guard runs before ``DeviceDriver.__init__``."""
        calls: list[int] = []
        original = rmd.DeviceDriver.__init__

        def recording_init(self, *args: Any, **kwargs: Any) -> None:
            calls.append(1)
            original(self, *args, **kwargs)

        monkeypatch.setattr(rmd.DeviceDriver, "__init__", recording_init)

        with pytest.raises(ValueError):
            rmd.ReachyMiniDriver(host="127.0.0.1/foo")
        assert calls == []

        rmd.ReachyMiniDriver(host="bot.local")
        assert calls == [1]

    def test_a_usable_host_still_probes_the_daemon(self, rmd, monkeypatch):
        """Control: the probe the guard protects still runs unchanged."""
        urls: list[str] = []

        def spy(req, body=None, timeout=None, **kwargs):
            urls.append(req.full_url)
            raise urllib.error.URLError("test: never dialed")

        class _FakeZenoh:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def start(self, **kwargs: Any) -> None:
                return None

        monkeypatch.setattr(urllib.request, "urlopen", spy)
        monkeypatch.setattr(rmd, "ZenohLink", _FakeZenoh)

        asyncio.run(rmd.ReachyMiniDriver(host="bot.local", api_port=9001).connect())
        assert urls == ["http://bot.local:9001/api/daemon/status"]


def _exported_host_constructors(source: str, exported: list[str]) -> dict[str, list[str]]:
    """Map each exported class to the host-ish ``__init__`` params it declares.

    Scoped to classes the package exports, for the same reason the port scan is:
    the hardware links are built only from an already-validated host, so making
    them re-check it would institutionalize a second copy of the rule.
    """
    found: dict[str, list[str]] = {}
    for node in walk_tree(parse_source(source)):
        if not isinstance(node, ast.ClassDef) or node.name not in exported:
            continue
        for member in node.body:
            if not isinstance(member, ast.FunctionDef) or member.name != "__init__":
                continue
            hosts = [
                arg.arg
                for arg in member.args.args + member.args.kwonlyargs
                if arg.arg == "host" or arg.arg.endswith("_host")
            ]
            if hosts:
                found[node.name] = hosts
    return found


def _validates_host(source: str, class_name: str) -> bool:
    """True when the class's ``__init__`` calls the shared host domain."""
    for node in walk_tree(parse_source(source)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "__init__":
                    return any(
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "dial_host_error"
                        for call in ast.walk(member)
                    )
    return False


class TestNoExportedDeviceConnectHostSurfaceDrifts:
    """Every exported driver that takes a host routes it through one domain.

    The port half already has this scan; it matches ``port``/``*_port`` only, so
    it could not see the host beside it. This is that scan for the other half.
    """

    def _surfaces(self) -> dict[str, tuple[Path, list[str]]]:
        package = Path(strands_robots.__file__).parent / "device_connect"
        exported = _exported_names(package / "__init__.py")
        surfaces: dict[str, tuple[Path, list[str]]] = {}
        for module in sorted(package.rglob("*.py")):
            source = module.read_text(encoding="utf-8")
            for class_name, hosts in _exported_host_constructors(source, exported).items():
                surfaces[class_name] = (module, hosts)
        return surfaces

    def test_the_scan_finds_the_known_host_surface(self):
        """Non-vacuity: a scan resolving elsewhere would report nothing."""
        assert {name: hosts for name, (_, hosts) in self._surfaces().items()} == {"ReachyMiniDriver": ["host"]}

    def test_every_exported_host_constructor_validates_it(self):
        """A future exported driver cannot dial a host unvalidated."""
        adrift = {
            name: hosts
            for name, (module, hosts) in self._surfaces().items()
            if not _validates_host(module.read_text(encoding="utf-8"), name)
        }
        assert adrift == {}, f"exported constructors taking a host without the shared domain: {adrift}"

    def test_the_scan_detects_a_planted_unguarded_host(self):
        """Meta: an empty result must mean clean sources, not a dead scanner."""
        planted = 'class Planted:\n    def __init__(self, host: str = "bot.local"):\n        self._h = host\n'
        assert _exported_host_constructors(planted, ["Planted"]) == {"Planted": ["host"]}
        assert not _validates_host(planted, "Planted")
