"""Smoke tests for :class:`strands_robots.policies.moveit2.MoveIt2Policy`.

Runs in-process against a stubbed ZMQ socket — no ROS 2, no live network.
Mirrors the pattern used by ``tests/policies/groot/test_zmq_wire_roundtrip.py``:
override ``client.socket.send`` / ``recv`` and msgpack-encode a fake
sidecar response.

Subtask 3 of issue #299. The acceptance criteria pin:

* :class:`MoveIt2Policy` is creatable via ``create_policy("moveit2", ...)``
  with the registered shorthand and the ``moveit`` alias.
* The wire request format matches the protocol the issue specifies
  (``joint_state``, ``target_pose`` / ``target_joints``, ``planning_group``,
  ``world_update``).
* Trajectory rows the sidecar returns unpack into per-step joint dicts.
* Validation rejects malformed goals up-front.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

msgpack = pytest.importorskip(
    "msgpack",
    reason="msgpack not installed - pip install 'strands-robots[moveit2]'",
)
zmq = pytest.importorskip(
    "zmq",
    reason="zmq not installed - pip install 'strands-robots[moveit2]'",
)

# E402: importorskip must execute before these imports to skip cleanly.
from strands_robots.policies import (  # noqa: E402
    Policy,
    create_policy,
    list_providers,
)
from strands_robots.policies.moveit2 import (  # noqa: E402
    MoveIt2InferenceClient,
    MoveIt2Policy,
    MsgSerializer,
)


def _capture_send_decode_recv(policy: MoveIt2Policy, response: dict) -> list[dict]:
    """Replace the client's send/recv with capturing stubs.

    Returns a list that gets populated with the *decoded* request dicts
    (one per ``call_endpoint`` round-trip). The recv stub returns
    ``response`` msgpack-packed.
    """
    sent: list[dict] = []

    def _capture_send(data: bytes) -> None:
        sent.append(MsgSerializer.from_bytes(data))

    packed = MsgSerializer.to_bytes(response)
    policy._client.socket.send = _capture_send  # type: ignore[assignment]
    policy._client.socket.recv = lambda: packed  # type: ignore[assignment]
    return sent


def _ok_trajectory_response(horizon: int = 4, ndof: int = 6) -> dict:
    """Construct a successful sidecar response with a synthetic trajectory."""
    trajectory = []
    for t in range(horizon):
        # [time, q0, q1, ...] - matches the wire protocol from issue #302.
        row = [float(t) * 0.1]
        for i in range(ndof):
            row.append(0.01 * (t + 1) * (i + 1))
        trajectory.append(row)
    return {"trajectory": trajectory, "success": True, "status": "ok"}


# ---------------------------------------------------------------------------
# MoveIt2InferenceClient
# ---------------------------------------------------------------------------


class TestMoveIt2InferenceClient:
    def test_construction_defaults_to_loopback(self):
        """Default host must be 127.0.0.1, not 0.0.0.0 (security baseline)."""
        client = MoveIt2InferenceClient()
        assert client.host == "127.0.0.1"
        assert client.port == 5556
        assert client.timeout_ms == 15000
        assert client.api_token is None

    def test_construction_with_api_token(self):
        client = MoveIt2InferenceClient(host="localhost", port=5556, api_token="secret")
        assert client.api_token == "secret"

    def test_api_token_warning_on_remote_host(self, caplog):
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="strands_robots.policies.moveit2.client"):
            MoveIt2InferenceClient(host="10.0.0.1", port=5556, api_token="tok")
        assert any("plaintext" in r.message for r in caplog.records)

    def test_no_warning_for_localhost_token(self, caplog):
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="strands_robots.policies.moveit2.client"):
            MoveIt2InferenceClient(host="127.0.0.1", port=5556, api_token="tok")
        assert not any("plaintext" in r.message for r in caplog.records)

    def test_call_endpoint_includes_api_token(self):
        client = MoveIt2InferenceClient(host="127.0.0.1", port=9999, api_token="mytoken")
        sent: list[dict] = []
        client.socket.send = lambda data: sent.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        client.call_endpoint("ping")
        assert sent[0]["api_token"] == "mytoken"

    def test_call_endpoint_without_api_token_omits_field(self):
        client = MoveIt2InferenceClient(host="127.0.0.1", port=9999)
        sent: list[dict] = []
        client.socket.send = lambda data: sent.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        client.call_endpoint("ping")
        assert "api_token" not in sent[0]

    def test_call_endpoint_raises_on_server_error(self):
        client = MoveIt2InferenceClient(host="127.0.0.1", port=9999)
        client.socket.send = MagicMock()
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"error": "no plan found"}))
        with pytest.raises(RuntimeError, match="Server error: no plan found"):
            client.call_endpoint("plan", {})

    def test_ping_returns_false_on_failure(self):
        client = MoveIt2InferenceClient(host="127.0.0.1", port=9999)
        client.socket.send = MagicMock(side_effect=Exception("timeout"))
        assert client.ping() is False

    def test_ping_returns_true_when_server_responds(self):
        """ping() reports True when the round-trip call_endpoint succeeds."""
        client = MoveIt2InferenceClient(host="127.0.0.1", port=9999)
        client.socket.send = MagicMock()
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        assert client.ping() is True

    def test_reconnect_closes_old_socket_and_connects_a_fresh_one(self):
        """reconnect() closes the current socket and builds a new connected one.

        A stale/timed-out REQ socket cannot be reused for another request, so
        the client must be able to swap in a fresh socket. The replacement must
        be a distinct object and must be re-connected to the same endpoint.
        """
        client = MoveIt2InferenceClient(host="127.0.0.1", port=9999)
        old_socket = client.socket
        # wraps= records the call for the assertion while still running the real
        # close(). A bare MagicMock would stub close() out entirely, orphaning a
        # live ZMQ socket on the still-open context - which raises
        # PytestUnraisableExceptionWarning when the GC finally reaps it (and, if
        # it is collected before the context, hangs context.term() forever).
        old_socket.close = MagicMock(wraps=old_socket.close)

        client.reconnect()

        old_socket.close.assert_called_once()
        # reconnect() must genuinely close the old socket, not merely call a
        # method named close - assert the socket is really shut, so a
        # regression that stops closing it (leaking the fd) is caught.
        assert old_socket.closed is True
        assert client.socket is not old_socket
        # New socket is usable for a round-trip (send/recv wired by _init_socket).
        client.socket.send = MagicMock()
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        assert client.ping() is True
        client._teardown()

    def test_reconnect_ignores_errors_closing_a_broken_socket(self):
        """A close() that raises must not stop reconnect from rebuilding.

        On a dead connection the old socket's close() can raise; reconnect must
        swallow it and still hand back a fresh, connected socket.
        """
        client = MoveIt2InferenceClient(host="127.0.0.1", port=9999)
        old_socket = client.socket
        real_close = old_socket.close
        old_socket.close = MagicMock(side_effect=RuntimeError("already dead"))

        client.reconnect()  # must not propagate

        assert client.socket is not old_socket
        # The stubbed close() raised, so the real socket is still open; close it
        # for real (and tear the client down) so no live ZMQ handle is orphaned.
        real_close()
        client._teardown()

    def test_teardown_swallows_socket_close_errors(self):
        """_teardown() is best-effort: a raising close()/term() never propagates.

        It runs from __del__ during interpreter shutdown, where a raised
        exception would be swallowed silently and could mask a resource leak,
        so the method itself must never raise.
        """
        client = MoveIt2InferenceClient(host="127.0.0.1", port=9999)
        real_close = client.socket.close
        real_term = client.context.term
        client.socket.close = MagicMock(side_effect=RuntimeError("boom"))
        client.context.term = MagicMock(side_effect=RuntimeError("boom"))
        client._teardown()  # must not raise
        # The stubbed close()/term() raised, so the real socket and context are
        # still live; release them for real so nothing is orphaned to the GC.
        real_close()
        real_term()

    def test_plan_helper_omits_optional_fields_when_unset(self):
        """plan() should not send ``target_pose`` / ``world_update`` keys when
        those are None — keeps the wire payload minimal and lets the
        sidecar use its own defaults."""
        client = MoveIt2InferenceClient(host="127.0.0.1", port=9999)
        sent: list[dict] = []
        client.socket.send = lambda data: sent.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(
            return_value=MsgSerializer.to_bytes({"trajectory": [], "success": True, "status": "ok"})
        )
        client.plan(joint_state=[0.0] * 6, planning_group="arm", target_joints={"j0": 0.5})
        payload = sent[0]["data"]
        assert "target_pose" not in payload
        assert "world_update" not in payload
        assert payload["target_joints"] == {"j0": 0.5}
        assert payload["planning_group"] == "arm"


# ---------------------------------------------------------------------------
# MoveIt2Policy - construction & registry
# ---------------------------------------------------------------------------


class TestMoveIt2PolicyConstruction:
    def test_provider_name(self):
        p = MoveIt2Policy(host="127.0.0.1", port=19999)
        assert p.provider_name == "moveit2"

    def test_does_not_require_images(self):
        """Planner-style policies must skip camera rendering (#300 contract)."""
        p = MoveIt2Policy(host="127.0.0.1", port=19999)
        assert p.requires_images is False

    def test_subclass_of_policy_abc(self):
        """Pin the inheritance contract from issue #300."""
        p = MoveIt2Policy(host="127.0.0.1", port=19999)
        assert isinstance(p, Policy)

    def test_silent_unknown_kwargs(self, caplog):
        """Per #300: providers MUST ignore unknown kwargs rather than raising."""
        # No exception raised on unknown kwarg.
        p = MoveIt2Policy(
            host="127.0.0.1",
            port=19999,
            future_kwarg_we_dont_know_about="ignore me",
        )
        assert p.host == "127.0.0.1"

    def test_create_policy_by_canonical_name(self):
        p = create_policy("moveit2", host="127.0.0.1", port=19999)
        assert isinstance(p, MoveIt2Policy)
        assert p.host == "127.0.0.1"
        assert p.port == 19999

    def test_create_policy_by_moveit_alias(self):
        p = create_policy("moveit", host="127.0.0.1", port=19999)
        assert isinstance(p, MoveIt2Policy)

    def test_listed_in_providers(self):
        providers = list_providers()
        assert "moveit2" in providers
        # ``moveit`` is an alias resolved at create_policy() time, not a
        # canonical name listed by ``list_providers()``. The alias path
        # is covered separately in ``test_create_policy_by_moveit_alias``.

    def test_api_token_env_fallback(self, monkeypatch):
        """Falls back to ``MOVEIT2_API_TOKEN`` env var when not provided."""
        monkeypatch.setenv("MOVEIT2_API_TOKEN", "env-token")
        p = MoveIt2Policy(host="127.0.0.1", port=19999)
        assert p._client.api_token == "env-token"

    def test_explicit_api_token_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MOVEIT2_API_TOKEN", "env-token")
        p = MoveIt2Policy(host="127.0.0.1", port=19999, api_token="explicit")
        assert p._client.api_token == "explicit"


# ---------------------------------------------------------------------------
# Validation - reject malformed goals up-front
# ---------------------------------------------------------------------------


class TestMoveIt2PolicyValidation:
    def _make_policy(self) -> MoveIt2Policy:
        return MoveIt2Policy(host="127.0.0.1", port=19999)

    def test_missing_target_raises(self):
        """Neither target_pose nor target_joints -> ValueError."""
        p = self._make_policy()
        with pytest.raises(ValueError, match="target_pose|target_joints"):
            asyncio.run(p.get_actions({"observation.state": [0.0] * 6}, "instruction"))

    def test_target_pose_wrong_length_rejected(self):
        p = self._make_policy()
        with pytest.raises(ValueError, match="7 elements"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_pose=[0.0, 0.0, 0.0],  # only 3 elements
                )
            )

    def test_target_pose_nan_rejected(self):
        p = self._make_policy()
        with pytest.raises(ValueError, match="finite"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_pose=[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, float("nan")],
                )
            )

    def test_target_joints_non_dict_rejected(self):
        p = self._make_policy()
        with pytest.raises(ValueError, match="must be a dict"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_joints=[0.5, 1.0],  # list instead of dict
                )
            )

    def test_target_joints_bad_key_rejected(self):
        """Joint names with shell metacharacters rejected up-front."""
        p = self._make_policy()
        with pytest.raises(ValueError, match="must match"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_joints={"j0; rm -rf": 0.5},
                )
            )

    def test_target_joints_inf_rejected(self):
        p = self._make_policy()
        with pytest.raises(ValueError, match="finite"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_joints={"j0": float("inf")},
                )
            )

    def test_planning_group_bad_chars_rejected(self):
        p = self._make_policy()
        with pytest.raises(ValueError, match="planning_group"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_joints={"j0": 0.5},
                    planning_group="arm; rm -rf /",
                )
            )


# ---------------------------------------------------------------------------
# Wire round-trip - end-to-end against stubbed ZMQ
# ---------------------------------------------------------------------------


class TestMoveIt2PolicyWireRoundTrip:
    def _make_policy(self) -> MoveIt2Policy:
        return MoveIt2Policy(
            host="127.0.0.1",
            port=19999,
            planning_group="arm",
        )

    def test_request_payload_has_canonical_schema(self):
        """The msgpack payload sent to the sidecar contains the keys the
        issue #302 wire protocol specifies: joint_state, target_pose,
        planning_group. ``options`` / ``api_token`` envelopes match the
        groot client behaviour."""
        p = self._make_policy()
        sent = _capture_send_decode_recv(p, _ok_trajectory_response())
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                "ignore me",
                target_pose=[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        assert len(sent) == 1
        assert sent[0]["endpoint"] == "plan"
        payload = sent[0]["data"]
        assert payload["joint_state"] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        assert payload["target_pose"] == [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
        assert payload["planning_group"] == "arm"
        assert "target_joints" not in payload  # not provided -> omitted

    def test_target_joints_path(self):
        """target_joints flows through cleanly."""
        p = self._make_policy()
        sent = _capture_send_decode_recv(p, _ok_trajectory_response())
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_joints={"shoulder_pan": 0.5, "elbow": -0.3},
            )
        )
        payload = sent[0]["data"]
        assert payload["target_joints"] == {"shoulder_pan": 0.5, "elbow": -0.3}

    def test_world_update_passthrough(self):
        """world_update is forwarded as-is (sidecar defines schema)."""
        p = self._make_policy()
        sent = _capture_send_decode_recv(p, _ok_trajectory_response())
        update = {"depth_topic": "/camera/depth", "stamp": 1234567890}
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_joints={"j0": 0.5},
                world_update=update,
            )
        )
        assert sent[0]["data"]["world_update"] == update

    def test_planning_group_per_call_override(self):
        """``planning_group`` kwarg overrides the constructor default."""
        p = MoveIt2Policy(host="127.0.0.1", port=19999, planning_group="arm")
        sent = _capture_send_decode_recv(p, _ok_trajectory_response())
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_joints={"j0": 0.5},
                planning_group="left_arm",
            )
        )
        assert sent[0]["data"]["planning_group"] == "left_arm"

    def test_trajectory_unpacks_to_per_step_dicts(self):
        """The sidecar's ``[[t, q0, q1, ...], ...]`` rows unpack into a
        list of per-step joint dicts. Time column is dropped — the
        runner schedules the timing."""
        p = self._make_policy()
        p.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5"])
        _capture_send_decode_recv(p, _ok_trajectory_response(horizon=4, ndof=6))

        actions = asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_pose=[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        assert len(actions) == 4
        # Per-step dict: 6 joints, no time column.
        for step in actions:
            assert set(step.keys()) == {"j0", "j1", "j2", "j3", "j4", "j5"}
            assert all(isinstance(v, float) for v in step.values())

    def test_trajectory_falls_back_to_positional_keys(self):
        """When ``set_robot_state_keys`` is unset, fall back to ``joint_<i>``."""
        p = self._make_policy()
        # Don't call set_robot_state_keys.
        _capture_send_decode_recv(p, _ok_trajectory_response(horizon=2, ndof=3))

        actions = asyncio.run(
            p.get_actions(
                {"observation.state": [0.0, 0.0, 0.0]},
                "",
                target_joints={"j0": 0.5},
            )
        )
        assert len(actions) == 2
        for step in actions:
            assert set(step.keys()) == {"joint_0", "joint_1", "joint_2"}

    def test_empty_trajectory_returns_empty_list(self):
        p = self._make_policy()
        _capture_send_decode_recv(p, {"trajectory": [], "success": True, "status": "ok"})
        actions = asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_joints={"j0": 0.5},
            )
        )
        assert actions == []

    def test_failed_plan_raises_runtime_error(self):
        """``success=False`` from the sidecar surfaces as a RuntimeError
        with status / goal context for debugging."""
        p = self._make_policy()
        _capture_send_decode_recv(
            p,
            {"trajectory": [], "success": False, "status": "no_collision_free_path"},
        )
        with pytest.raises(RuntimeError, match="no_collision_free_path"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_pose=[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                )
            )

    def test_reset_forwards_to_server(self):
        """Pin the reset() round-trip behaviour: server sees the seed."""
        p = self._make_policy()
        sent = _capture_send_decode_recv(p, {"status": "ok"})
        p.reset(seed=42)
        assert sent[0]["endpoint"] == "reset"
        assert sent[0]["data"] == {"options": {"seed": 42}}

    def test_reset_swallows_server_errors(self):
        """reset() is best-effort; server errors must not propagate."""
        p = self._make_policy()
        # Mock send to raise so call_endpoint propagates an exception.
        p._client.socket.send = MagicMock(side_effect=Exception("connection refused"))
        # Should not raise.
        p.reset(seed=42)


# ---------------------------------------------------------------------------
# Policy ABC contract — same shape as MockPolicy
# ---------------------------------------------------------------------------


class TestPolicyContractParity:
    """Mock + MoveIt2 must pass the same Policy-shape contract.

    Pins the issue #300 ABC contract for non-VLA providers so a future
    refactor that breaks one cannot pass while breaking the other. cuRobo
    (subtask 2 of #299) will join this list when it lands.
    """

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: create_policy("mock"),
            lambda: create_policy("moveit2", host="127.0.0.1", port=19999),
        ],
    )
    def test_provider_is_policy_subclass(self, factory):
        assert isinstance(factory(), Policy)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: create_policy("mock"),
            lambda: create_policy("moveit2", host="127.0.0.1", port=19999),
        ],
    )
    def test_provider_has_provider_name(self, factory):
        p = factory()
        assert isinstance(p.provider_name, str)
        assert p.provider_name  # non-empty

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: create_policy("mock"),
            lambda: create_policy("moveit2", host="127.0.0.1", port=19999),
        ],
    )
    def test_provider_set_robot_state_keys_is_no_raise(self, factory):
        p = factory()
        # Both implementations accept the call shape from #300; mock
        # stores the list, moveit2 stores the list for trajectory
        # unpacking. Neither raises.
        p.set_robot_state_keys(["j0", "j1", "j2"])

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: create_policy("mock"),
            lambda: create_policy("moveit2", host="127.0.0.1", port=19999),
        ],
    )
    def test_provider_requires_images_is_false_for_planners(self, factory):
        """Both providers consume joint state only - skip camera rendering."""
        assert factory().requires_images is False

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: create_policy("mock"),
            lambda: create_policy("moveit2", host="127.0.0.1", port=19999),
        ],
    )
    def test_provider_reset_is_no_raise(self, factory):
        """reset() is best-effort and must not raise on the default path."""
        p = factory()
        # MoveIt2 forwards to a (stubbed-absent) server; the client send
        # would fail, but reset() catches and logs.
        if isinstance(p, MoveIt2Policy):
            p._client.socket.send = MagicMock(side_effect=Exception("offline"))
        p.reset(seed=0)


# ---------------------------------------------------------------------------
# Goal-validation and trajectory-decode edge cases
#
# These drive the remaining defence-in-depth branches in get_actions through
# the public API: the joint-state extraction fail-soft paths, malformed goal
# rejection (non-iterable / non-numeric / wrong-type), and the empty-row skip
# in trajectory unpacking. They assert on the observable contract (raised
# error, forwarded payload, returned actions), not on internal state.
# ---------------------------------------------------------------------------


class TestMoveIt2JointStateExtraction:
    def _make_policy(self) -> MoveIt2Policy:
        return MoveIt2Policy(host="127.0.0.1", port=19999, planning_group="arm")

    def test_missing_state_forwards_none_joint_state(self):
        """No ``observation.state`` -> ``joint_state`` omitted so the sidecar
        falls back to its own state estimate (not a fabricated zero pose)."""
        p = self._make_policy()
        sent = _capture_send_decode_recv(p, _ok_trajectory_response())
        asyncio.run(p.get_actions({}, "", target_joints={"j0": 0.5}))
        assert sent[0]["data"]["joint_state"] is None

    def test_numpy_state_is_serialised_to_plain_floats(self):
        """A numpy ``observation.state`` is converted via ``tolist`` so the
        msgpack payload carries plain Python floats (no numpy on the wire)."""
        import numpy as np

        p = self._make_policy()
        sent = _capture_send_decode_recv(p, _ok_trajectory_response())
        asyncio.run(
            p.get_actions(
                {"observation.state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])},
                "",
                target_joints={"j0": 0.5},
            )
        )
        joint_state = sent[0]["data"]["joint_state"]
        assert joint_state == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert all(isinstance(x, float) for x in joint_state)

    def test_non_numeric_state_degrades_to_none(self, caplog):
        """A non-numeric ``observation.state`` is fail-soft: the start config
        is dropped (sidecar uses its own estimate) rather than crashing the
        plan call, and the failure is logged."""
        p = self._make_policy()
        sent = _capture_send_decode_recv(p, _ok_trajectory_response())
        import logging

        with caplog.at_level(logging.WARNING):
            asyncio.run(
                p.get_actions(
                    {"observation.state": ["not", "a", "number"]},
                    "",
                    target_joints={"j0": 0.5},
                )
            )
        assert sent[0]["data"]["joint_state"] is None
        assert "failed to extract joint_state" in caplog.text


class TestMoveIt2GoalTypeRejection:
    def _make_policy(self) -> MoveIt2Policy:
        return MoveIt2Policy(host="127.0.0.1", port=19999, planning_group="arm")

    def test_non_iterable_target_pose_rejected(self):
        """A scalar ``target_pose`` (can't be list()-ed) is rejected with a
        type-named message rather than a downstream TypeError."""
        p = self._make_policy()
        with pytest.raises(ValueError, match="7-element list, got int"):
            asyncio.run(p.get_actions({"observation.state": [0.0] * 6}, "", target_pose=7))

    def test_non_numeric_target_pose_element_rejected(self):
        """A non-numeric pose element is rejected up-front (not float()-ed at
        the sidecar)."""
        p = self._make_policy()
        with pytest.raises(ValueError, match=r"target_pose\[6\] must be a number, got NoneType"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_pose=[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, None],
                )
            )

    def test_non_numeric_target_joints_value_rejected(self):
        """A non-numeric joint value is rejected with the joint name in the
        message."""
        p = self._make_policy()
        with pytest.raises(ValueError, match="must be a number"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_joints={"j0": None},
                )
            )

    def test_non_string_planning_group_rejected(self):
        """A non-string ``planning_group`` is rejected before it reaches the
        sidecar's parameter interpolation."""
        p = self._make_policy()
        with pytest.raises(ValueError, match="planning_group must be a str, got int"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_joints={"j0": 0.5},
                    planning_group=123,
                )
            )


class TestMoveIt2TrajectoryDecode:
    def _make_policy(self) -> MoveIt2Policy:
        return MoveIt2Policy(host="127.0.0.1", port=19999, planning_group="arm")

    def test_empty_trajectory_row_is_skipped(self):
        """An empty waypoint row from the sidecar is skipped, not emitted as
        an empty action dict that the runner would have to special-case."""
        p = self._make_policy()
        response = {
            "trajectory": [[0.0, 0.1, 0.2], [], [0.1, 0.3, 0.4]],
            "success": True,
            "status": "ok",
        }
        _capture_send_decode_recv(p, response)
        actions = asyncio.run(p.get_actions({"observation.state": [0.0, 0.0]}, "", target_joints={"j0": 0.5}))
        # Two non-empty rows -> two actions; the empty row produced nothing.
        assert len(actions) == 2
        assert all(set(step.keys()) == {"joint_0", "joint_1"} for step in actions)


class TestClientTeardownNonBlocking:
    """Teardown must not block on a request queued to a dead sidecar.

    A REQ socket connected to an unreachable server buffers the outgoing
    request internally. With the default (infinite) ZMQ linger, ``close()``
    (and the ``context.term()`` that follows) blocks forever waiting to flush
    that undelivered request - which stalls the GC that drives ``__del__`` and
    interpreter shutdown. The client pins ``LINGER=0`` at socket creation so
    the buffered request is discarded and teardown returns promptly.
    """

    @staticmethod
    def _dead_port() -> int:
        """Return a TCP port with nothing listening on it."""
        import socket as _socket

        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def test_socket_linger_is_zero(self):
        client = MoveIt2InferenceClient(host="127.0.0.1", port=self._dead_port())
        try:
            assert client.socket.getsockopt(zmq.LINGER) == 0
        finally:
            client._teardown()

    def test_teardown_bounded_with_queued_request_to_dead_server(self):
        import threading

        client = MoveIt2InferenceClient(host="127.0.0.1", port=self._dead_port())
        # Queue a request that can never be delivered (no peer). send() returns
        # immediately; the bytes sit in the socket's outgoing buffer.
        client.socket.send(b"never-delivered")

        done = threading.Event()

        def _run() -> None:
            client._teardown()
            done.set()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        # Pre-fix (infinite linger) this blocks forever; post-fix it is instant.
        assert done.wait(timeout=10.0), "client teardown blocked on a queued request to a dead server"
