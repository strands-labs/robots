"""What address, what port and what patience a single-node elastic launch gets.

These are pure: they never construct a store or spawn a worker, so they run in
milliseconds. The address and port a launch rendezvouses on are derived rather than
left to torch, because torch's own fallbacks for both are addresses no peer can
reach - ``socket.getfqdn()`` for ``MASTER_ADDR`` and ``localhost:0`` for the
endpoint. Both resolutions happen inside libtorch's C++ socket code, so a wrong
value there is a silent wait that no Python-level timeout can interrupt; the only
way to test the decision is to make it in Python first, which is what these
functions exist for.

``fqdn`` and ``port_picker`` are injected throughout, so no assertion here depends
on the runner's own DNS or on a real bind.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from strands_robots.training._inproc import (
    DEFAULT_RDZV_TIMEOUT_S,
    LOCAL_ADDR_ENV,
    RDZV_TIMEOUT_ENV,
    elastic_launch_callable,
    free_local_port,
    launch_local_addr,
    looks_like_reverse_dns,
    rdzv_timeout_s,
    rendezvous_endpoint,
)


def _worker() -> int:
    return 0


class TestRendezvousEndpoint:
    """The endpoint is a dialable address, never port 0 and never an ambiguous name."""

    def test_an_explicit_endpoint_always_wins(self) -> None:
        assert rendezvous_endpoint("head-node:29500", 4) == "head-node:29500"
        assert rendezvous_endpoint("head-node:29500", 1) == "head-node:29500"

    def test_a_single_node_launch_gets_a_concrete_loopback_port(self) -> None:
        assert rendezvous_endpoint("", 1, port_picker=lambda: 45678) == "127.0.0.1:45678"

    def test_never_port_zero_and_never_the_ambiguous_localhost(self) -> None:
        # Port 0 is not an address a client can dial, and ``localhost`` can resolve to
        # both ::1 and 127.0.0.1, which lets the store server and its client bind
        # different stacks and miss each other inside libtorch.
        host, _, port = rendezvous_endpoint("", 1).rpartition(":")
        assert host == "127.0.0.1"
        assert int(port) > 0

    def test_a_multi_node_launch_without_an_endpoint_is_refused_with_the_reason(self) -> None:
        with pytest.raises(ValueError, match="rdzv_endpoint"):
            rendezvous_endpoint("", 3)

    def test_the_picked_port_is_actually_free(self) -> None:
        port = free_local_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))  # would raise if anything still held it


class TestRdzvTimeout:
    """The bound handed to torch, and what an operator's typo may not do to it."""

    def test_default_when_unset(self) -> None:
        assert rdzv_timeout_s({}) == DEFAULT_RDZV_TIMEOUT_S

    def test_an_operator_can_raise_it_for_a_slow_cluster(self) -> None:
        assert rdzv_timeout_s({RDZV_TIMEOUT_ENV: "900"}) == 900

    @pytest.mark.parametrize("junk", ["", "soon", "0", "-5", "None", "0x10"])
    def test_junk_and_non_positive_fall_back_rather_than_disabling_the_bound(self, junk: str) -> None:
        # An unparseable or non-positive value must not be able to restore the
        # unbounded wait this bound exists to prevent.
        assert rdzv_timeout_s({RDZV_TIMEOUT_ENV: junk}) == DEFAULT_RDZV_TIMEOUT_S

    @pytest.mark.parametrize("value", ["inf", "-inf", "1e999", "nan"])
    def test_a_non_finite_value_falls_back_instead_of_raising(self, value: str) -> None:
        # ``int(float("inf"))`` raises OverflowError and ``int(float("nan"))`` raises
        # ValueError, so reading this env var must not convert first and ask questions
        # after: an env typo has to yield the default, not a traceback out of a
        # training launch.
        assert rdzv_timeout_s({RDZV_TIMEOUT_ENV: value}) == DEFAULT_RDZV_TIMEOUT_S


class TestLocalAddr:
    """The address published as ``MASTER_ADDR``.

    Left to torch this is ``local_addr or socket.getfqdn()``. A host whose
    ``getfqdn()`` answers with a reverse-DNS PTR name publishes a name with no
    forward lookup, so the worker store's client dials an address it can never
    resolve and libtorch retries inside its C++ socket code - a run parked on
    "Rendezvous'ing worker group" with no error and no timeout that can reach it.
    """

    def test_a_single_node_launch_is_pinned_to_loopback(self) -> None:
        # Even when this host's own name is the unresolvable kind.
        assert launch_local_addr(1, env={}, fqdn=lambda: "1.0.0.0.ip6.arpa") == "127.0.0.1"

    def test_a_broken_fqdn_cannot_reach_a_single_node_launch(self) -> None:
        def _boom() -> str:
            raise OSError("no resolution here")

        assert launch_local_addr(1, env={}, fqdn=_boom) == "127.0.0.1"

    def test_an_explicit_address_wins_everywhere(self) -> None:
        assert launch_local_addr(1, "10.0.0.7", env={}) == "10.0.0.7"
        assert launch_local_addr(4, "10.0.0.7", env={}) == "10.0.0.7"

    def test_the_operator_can_override_by_env(self) -> None:
        assert launch_local_addr(4, env={LOCAL_ADDR_ENV: " 10.0.0.9 "}) == "10.0.0.9"

    def test_a_multi_node_launch_keeps_torchs_own_resolution(self) -> None:
        # The address must be reachable from the OTHER nodes, so a value guessed here
        # would be worse than letting torch resolve one.
        assert launch_local_addr(4, env={}, fqdn=lambda: "head.cluster.local") is None

    def test_a_multi_node_launch_with_a_reverse_dns_fqdn_is_warned_about(self, caplog: Any) -> None:
        with caplog.at_level("WARNING"):
            assert launch_local_addr(4, env={}, fqdn=lambda: "1.0.0.0.0.ip6.arpa") is None
        assert LOCAL_ADDR_ENV in caplog.text
        assert "hang" in caplog.text, "a silent hang deserves to be named as one"

    @pytest.mark.parametrize(
        "name,is_reverse",
        [
            ("1.0.0.0.0.0.0.0.ip6.arpa", True),
            ("1.0.0.0.0.0.0.0.ip6.arpa.", True),
            ("4.3.2.1.in-addr.arpa", True),
            # The zone apex itself, in any case: not a name anyone can dial either.
            ("IP6.ARPA", True),
            ("head.cluster.local", False),
            ("127.0.0.1", False),
            # A hostname that merely starts with the zone's letters is a real name.
            ("arpa-node.example.com", False),
        ],
    )
    def test_reverse_dns_names_are_recognised(self, name: str, is_reverse: bool) -> None:
        assert looks_like_reverse_dns(name) is is_reverse


class TestTheLaunchCarriesTheseDecisions:
    """The derived values reach ``LaunchConfig``, not just the helpers that make them.

    A helper that returns the right address is worth nothing if the launch config is
    still built from torch's defaults, so the config is captured and read here rather
    than trusting that the wiring exists.
    """

    def _captured_config(self, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
        from torch.distributed.launcher.api import elastic_launch as real_launch

        seen: dict[str, Any] = {}

        def _capture(config: Any, entrypoint: Any) -> Any:
            seen["config"] = config
            return lambda *a, **k: {0: 0}

        monkeypatch.setattr("torch.distributed.launcher.api.elastic_launch", _capture)
        assert real_launch is not _capture  # the patch replaced something real
        elastic_launch_callable(_worker, nproc_per_node=1, **kwargs)
        return seen["config"]

    def test_the_endpoint_is_a_concrete_loopback_port_not_localhost_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = self._captured_config(monkeypatch)
        host, _, port = config.rdzv_endpoint.rpartition(":")
        assert host == "127.0.0.1", config.rdzv_endpoint
        assert int(port) > 0, config.rdzv_endpoint

    def test_master_addr_is_pinned_rather_than_resolved_from_this_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # None here is what makes torch fall back to socket.getfqdn().
        config = self._captured_config(monkeypatch)
        assert config.local_addr == "127.0.0.1"

    def test_an_explicit_local_addr_is_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = self._captured_config(monkeypatch, local_addr="10.0.0.7")
        assert config.local_addr == "10.0.0.7"

    def test_every_rendezvous_phase_carries_the_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The c10d handler reads join_timeout, its store reads read_timeout, and the
        # static backend reads timeout; a phase left at its default is a wait that
        # outlives the bound.
        configs = self._captured_config(monkeypatch).rdzv_configs
        for key in ("timeout", "read_timeout", "join_timeout"):
            assert configs[key] == DEFAULT_RDZV_TIMEOUT_S, (key, configs)

    def test_an_explicit_endpoint_still_reaches_the_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = self._captured_config(monkeypatch, rdzv_endpoint="head-node:29500")
        assert config.rdzv_endpoint == "head-node:29500"
