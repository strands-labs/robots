"""A policy provider refuses a service port it cannot dial.

Four providers reach a policy service over TCP and build the endpoint by
interpolating a caller-supplied ``port``: ``groot`` and ``moveit2`` into
``tcp://<host>:<port>`` (ZMQ), ``cosmos3`` into ``ws://<host>:<port>``, and
``lerobot_async`` into a gRPC target. Every one of those transports connects
lazily, so the socket does not reject a port outside the 16-bit range - it
accepts it and the request fails much later as an unreachable service, naming
the server rather than the port that could never have addressed it.

:func:`~strands_robots.utils.tcp_port_error` is the shared domain the agent tool
that *starts* the GR00T service (``gr00t_inference``) and the mesh bridges
already validate against, and its docstring states the invariant these
providers broke: the same port cannot be refused onto a service by one surface
and accepted by the next. These tests pin the refusal per provider, pin that it
precedes any transport construction, pin the values each provider still accepts,
and pin the branch scoping - a port a call never dials is not validated.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies import create_policy
from strands_robots.utils import tcp_port_error

# Ports no TCP transport can address. ``0`` asks the kernel for an ephemeral
# port instead of naming one; ``65536``/``99999`` are outside the 16-bit space;
# ``2.7`` and ``"5555"`` are not port indices; ``True`` is an ``int`` subclass
# that would land on privileged port 1; the rest cannot be interpolated into an
# endpoint at all.
UNUSABLE_PORTS: list[Any] = [
    0,
    -1,
    65536,
    99999,
    2.7,
    "5555",
    True,
    False,
    float("nan"),
    float("inf"),
    None,
    [5555],
]

#: Ports every provider must keep accepting: the range boundaries and the
#: providers' own defaults.
USABLE_PORTS: list[Any] = [1, 5555, 5556, 8000, 8080, 65535]

#: A NumPy integer is not a Python ``int`` subclass, so the shared domain
#: refuses it. It is listed separately because the point is agreement rather
#: than acceptance: the providers must give the same answer as every other
#: surface, not invent a wider or narrower one of their own.
DOMAIN_REFUSED_NON_INT: list[Any] = [np.int64(8080), np.float64(8080.0)]


def _groot(port: Any) -> Any:
    return create_policy("groot", port=port, data_config="so100_dualcam")


def _moveit2(port: Any) -> Any:
    return create_policy("moveit2", port=port)


def _cosmos3(port: Any) -> Any:
    return create_policy("cosmos3", port=port, embodiment="droid")


def _lerobot_async(port: Any) -> Any:
    return create_policy("lerobot_async", port=port, policy_type="act", pretrained_name_or_path="org/ckpt")


#: ``(provider label, constructor, class name, optional dependency)``. The
#: refusal happens before any transport is built, so a rejected port needs no
#: optional dependency; only the accepted-value controls dial and therefore
#: import one.
PROVIDERS: list[tuple[str, Any, str, str | None]] = [
    ("groot", _groot, "Gr00tPolicy", "zmq"),
    ("moveit2", _moveit2, "MoveIt2Policy", "zmq"),
    ("cosmos3", _cosmos3, "Cosmos3Policy", None),
    ("lerobot_async", _lerobot_async, "LerobotAsyncPolicy", None),
]

_IDS = [p[0] for p in PROVIDERS]


class TestUnusablePortIsRefused:
    """Every provider refuses a port it could only interpolate, not dial."""

    @pytest.mark.parametrize(("label", "build", "cls_name", "dep"), PROVIDERS, ids=_IDS)
    @pytest.mark.parametrize("port", UNUSABLE_PORTS, ids=repr)
    def test_provider_refuses(self, label: str, build: Any, cls_name: str, dep: str | None, port: Any) -> None:
        """The refusal is a ValueError naming the class, the parameter and the value."""
        with pytest.raises(ValueError) as excinfo:
            build(port)
        message = str(excinfo.value)
        assert cls_name in message, message
        assert "invalid port" in message, message
        assert repr(port) in message, message
        assert "1-65535" in message, message

    @pytest.mark.parametrize(("label", "build", "cls_name", "dep"), PROVIDERS, ids=_IDS)
    def test_refusal_matches_the_shared_domain(self, label: str, build: Any, cls_name: str, dep: str | None) -> None:
        """A provider's verdict equals the shared domain's for every probe value.

        The domain is what keeps the surfaces in step, so the test compares
        against it rather than restating a range per provider.
        """
        for port in UNUSABLE_PORTS:
            assert tcp_port_error(port, "port", cls_name) is not None, port
            with pytest.raises(ValueError):
                build(port)
        for port in USABLE_PORTS:
            assert tcp_port_error(port, "port", cls_name) is None, port
        for port in DOMAIN_REFUSED_NON_INT:
            assert tcp_port_error(port, "port", cls_name) is not None, port
            with pytest.raises(ValueError):
                build(port)


class TestRefusalPrecedesTheTransport:
    """A refused port never reaches a socket, context or channel."""

    @pytest.mark.parametrize(
        ("module_path", "transport_attr", "build"),
        [
            ("strands_robots.policies.groot.policy", "Gr00tInferenceClient", _groot),
            ("strands_robots.policies.moveit2.policy", "MoveIt2InferenceClient", _moveit2),
            ("strands_robots.policies.cosmos3.policy", "Cosmos3WebsocketClient", _cosmos3),
        ],
        ids=["groot", "moveit2", "cosmos3"],
    )
    def test_transport_is_never_constructed(
        self, monkeypatch: pytest.MonkeyPatch, module_path: str, transport_attr: str, build: Any
    ) -> None:
        """Constructing the transport is made fatal, so reaching it fails the test."""
        import importlib

        module = importlib.import_module(module_path)

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"{transport_attr} was constructed for a refused port")

        monkeypatch.setattr(module, transport_attr, explode)
        with pytest.raises(ValueError, match="invalid port"):
            build(99999)


class TestUsablePortIsAccepted:
    """The change is additive: nothing a provider could dial is now refused."""

    @pytest.mark.parametrize(("label", "build", "cls_name", "dep"), PROVIDERS, ids=_IDS)
    @pytest.mark.parametrize("port", USABLE_PORTS, ids=repr)
    def test_provider_accepts(self, label: str, build: Any, cls_name: str, dep: str | None, port: Any) -> None:
        if dep is not None:
            pytest.importorskip(dep, reason=f"{dep} not installed - the {label} transport cannot be built")
        policy = build(port)
        assert type(policy).__name__ == cls_name


class TestOnlyTheDialedPortIsValidated:
    """A port a call never reads is not refused - the guard is scoped."""

    def test_groot_local_mode_ignores_the_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``model_path`` selects local inference, which dials nothing."""
        from strands_robots.policies.groot.policy import Gr00tPolicy

        monkeypatch.setattr(Gr00tPolicy, "_load_local_policy", lambda self, *a, **k: None)
        monkeypatch.setattr(Gr00tPolicy, "_init_mappings", lambda self: None)
        policy = Gr00tPolicy(model_path="/tmp/checkpoint", port=99999)
        assert policy._mode == "local"

    def test_cosmos3_injected_client_owns_its_address(self) -> None:
        """An injected client already holds an endpoint, so ``port`` is inert."""
        from strands_robots.policies.cosmos3 import Cosmos3Policy

        sentinel = object()
        # An inert stand-in: what is under test is that the port branch is
        # skipped, not anything the client does.
        policy = Cosmos3Policy(embodiment="droid", port=99999, client=sentinel)  # type: ignore[arg-type]
        assert policy._client is sentinel

    def test_cosmos3_diffusers_backend_dials_nothing(self) -> None:
        """The in-process backend has no service endpoint to address."""
        from strands_robots.policies.cosmos3 import Cosmos3Policy

        sentinel = object()
        policy = Cosmos3Policy(
            embodiment="droid",
            port=99999,
            backend="diffusers",
            diffusers_backend=sentinel,  # type: ignore[arg-type]  # inert stand-in: the port branch is what is under test
        )
        assert policy.backend == "diffusers"

    def test_lerobot_async_server_address_supersedes_the_port(self) -> None:
        """``server_address`` is the effective spelling when it is given."""
        from strands_robots.policies.lerobot_async import LerobotAsyncPolicy

        policy = LerobotAsyncPolicy(
            server_address="gpu-box:8080",
            port=99999,
            policy_type="act",
            pretrained_name_or_path="org/ckpt",
        )
        assert policy.server_address == "gpu-box:8080"


class TestTransportDoesNotRefuseItself:
    """Why the boundary guard is load-bearing rather than belt-and-braces."""

    def test_zmq_connect_accepts_a_port_outside_the_range(self) -> None:
        """ZMQ resolves the endpoint lazily, so the socket accepts port 99999.

        This is the premise the guard rests on: without it, the value is not
        refused anywhere - it becomes an endpoint that can never connect and
        surfaces as an inference timeout blamed on the server.
        """
        zmq = pytest.importorskip("zmq", reason="zmq not installed - pip install 'strands-robots[groot-service]'")
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect("tcp://localhost:99999")
            assert socket.getsockopt_string(zmq.LAST_ENDPOINT) == "tcp://localhost:99999"
        finally:
            socket.close()
            context.term()


def _policy_module_paths() -> list[Path]:
    """Every provider ``policy.py`` under the policies package.

    The scan root is derived from ``create_policy`` -- the package's own factory,
    so its defining module is inside the package by construction -- rather than
    from a path literal or a second import of the package itself. A root that
    resolved elsewhere would make the scan below silently empty, which is what
    the "these four classes were seen" assertion in
    :class:`TestNoProviderShipsAnUnguardedPort` exists to catch.
    """
    root = Path(inspect.getfile(create_policy)).parent
    return sorted(root.glob("*/policy.py"))


def _classes_taking_a_port(tree: ast.Module) -> list[ast.ClassDef]:
    """Class definitions whose ``__init__`` declares a ``port`` parameter."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            args = item.args
            names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
            if "port" in names:
                found.append(node)
    return found


def _calls_the_shared_domain(cls: ast.ClassDef) -> bool:
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "tcp_port_error"
        for node in ast.walk(cls)
    )


class TestNoProviderShipsAnUnguardedPort:
    """A fifth provider cannot repeat this by copying a sibling constructor."""

    def test_every_provider_taking_a_port_validates_it(self) -> None:
        offenders = []
        seen = []
        for path in _policy_module_paths():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in _classes_taking_a_port(tree):
                seen.append(cls.name)
                if not _calls_the_shared_domain(cls):
                    offenders.append(f"{path.name}::{cls.name}")
        assert set(_IDS) and {"Gr00tPolicy", "MoveIt2Policy", "Cosmos3Policy", "LerobotAsyncPolicy"} <= set(seen), seen
        assert offenders == [], (
            "these provider constructors accept a port without validating it against "
            f"strands_robots.utils.tcp_port_error: {offenders}"
        )

    def test_the_scan_detects_an_unguarded_constructor(self) -> None:
        """A planted omission is reported, so an empty result means clean sources."""
        planted = ast.parse(
            "class RoguePolicy:\n"
            "    def __init__(self, host='h', port=1234):\n"
            "        self.endpoint = f'tcp://{host}:{port}'\n"
        )
        classes = _classes_taking_a_port(planted)
        assert [c.name for c in classes] == ["RoguePolicy"]
        assert not _calls_the_shared_domain(classes[0])
