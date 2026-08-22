"""Both publishers on the cmd-topic byte cap report an over-cap payload.

The transport's ``low_pass_filter`` caps ``**/cmd`` and ``**/broadcast`` with
one rule (``strands_cmd_size_cap``, ingress and egress) while the command
validator admits a ``world_update`` four times that size
(:data:`~strands_robots.mesh.security.MAX_WORLD_UPDATE_BYTES` is 64 KiB against
a 16 KiB cap). A validator-valid command in that gap is dropped by the filter,
which cannot say why nothing arrived.

:meth:`~strands_robots.mesh.core.Mesh.send` pre-checked the encoded size and
answered with a structured error naming the cap.
:meth:`~strands_robots.mesh.core.Mesh.broadcast` publishes on the other key
expression of the same rule and did not, so an over-cap broadcast was handed to
the filter and returned the empty list a broadcast nobody answered also
returns -- indistinguishable, on a method ``robot_mesh`` exposes as a fleet-wide
action. Both now ask one helper, so the two topics cannot come to disagree
about the cap they share.
"""

import ast
import inspect
import json
import logging
import sys
import textwrap

import pytest

from strands_robots.mesh import core
from strands_robots.mesh import security as _security
from strands_robots.mesh._zenoh_config import cmd_bytes_cap

#: A validator-VALID execute command whose ``world_update`` sits in the gap
#: between the validator's bound and the transport's cap.
_OVER_CAP_CMD: dict[str, object] = {
    "action": "execute",
    "instruction": "go",
    "policy_provider": "mock",
    "world_update": {"blob": "x" * 20000},
}
_SMALL_CMD: dict[str, object] = {"action": "stop"}


def _mesh() -> core.Mesh:
    m = core.Mesh(robot=object(), peer_id="op")
    m._running = True
    return m


def _drive(m: core.Mesh, publisher: str, cmd: dict[str, object]) -> tuple[object, list[tuple[str, int]]]:
    """Run *publisher* with *cmd*, returning ``(result, what reached the wire)``."""
    reached: list[tuple[str, int]] = []
    m.publish = lambda key, payload: reached.append(  # type: ignore[method-assign]
        (key, len(json.dumps(payload).encode("utf-8")))
    )
    if publisher == "send":
        return m.send("r1", cmd, timeout=0.05), reached
    return m.broadcast(cmd, timeout=0.05), reached


class TestTheOverCapPayloadIsWithinTheValidatorsBound:
    """Premise: the fixture is a command the validator accepts."""

    def test_the_validator_admits_the_over_cap_command(self) -> None:
        # Without this the tests below would measure a validation refusal
        # rather than the size cap.
        _security.validate_command(dict(_OVER_CAP_CMD))

    def test_the_fixture_exceeds_the_cap_it_is_about(self) -> None:
        encoded = len(json.dumps(_OVER_CAP_CMD).encode("utf-8"))
        assert encoded > cmd_bytes_cap(), f"fixture is {encoded} bytes, cap is {cmd_bytes_cap()}"


class TestNeitherPublisherHandsAnOverCapPayloadToTheFilter:
    """The headline. ``send`` passes on both trees; ``broadcast`` is the fix."""

    @pytest.mark.parametrize("publisher", ["send", "broadcast"])
    def test_an_over_cap_command_does_not_reach_the_transport(self, publisher: str) -> None:
        m = _mesh()
        _result, reached = _drive(m, publisher, dict(_OVER_CAP_CMD))
        assert reached == [], (
            f"{publisher}() handed a {len(json.dumps(_OVER_CAP_CMD).encode('utf-8'))}-byte command to the "
            f"transport, whose cmd-topic rule caps it at {cmd_bytes_cap()} bytes and drops it silently: {reached}"
        )

    @pytest.mark.parametrize("publisher", ["send", "broadcast"])
    def test_the_reason_names_the_actual_size_the_cap_and_the_env_var(
        self, publisher: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        m = _mesh()
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
            result, _reached = _drive(m, publisher, dict(_OVER_CAP_CMD))
        # ``send`` reports in its return value, ``broadcast`` in the log --
        # each uses the shape its own return type already has for a
        # client-side rejection. Read whichever this publisher offers.
        reported = json.dumps(result) + " " + caplog.text
        cap = cmd_bytes_cap()
        for expected in (str(cap), "STRANDS_MESH_MAX_CMD_BYTES", "bytes"):
            assert expected in reported, f"{publisher}() did not name {expected!r}: {reported[:400]}"

    @pytest.mark.parametrize("publisher", ["send", "broadcast"])
    def test_the_refused_turn_leaves_no_pending_state(self, publisher: str) -> None:
        m = _mesh()
        _drive(m, publisher, dict(_OVER_CAP_CMD))
        assert m._pending == {}, m._pending
        assert m._responses == {}, m._responses
        assert m._expected_responders == {}, m._expected_responders


class TestABroadcastRejectionIsReportedTheWayThisMethodAlreadyReportsOne:
    """``broadcast`` returns ``list``, so its rejection slot is the log."""

    def test_the_over_cap_broadcast_is_logged_as_a_client_side_rejection(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        m = _mesh()
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
            result = m.broadcast(dict(_OVER_CAP_CMD), timeout=0.05)
        assert result == []
        assert "broadcast rejected client-side" in caplog.text, caplog.text

    def test_a_validation_rejection_still_uses_that_same_shape(self, caplog: pytest.LogCaptureFixture) -> None:
        # The wording this fix reuses is not new: an invalid command has been
        # reported this way. Both rejections must stay one shape.
        m = _mesh()
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
            result = m.broadcast({"action": "execute", "instruction": "go"}, timeout=0.05)
        assert result == []
        assert "broadcast rejected client-side" in caplog.text, caplog.text


class TestNothingElseChanges:
    """Controls -- each fails for a specific over-reaching fix."""

    @pytest.mark.parametrize("publisher", ["send", "broadcast"])
    def test_a_payload_under_the_cap_still_reaches_the_transport(self, publisher: str) -> None:
        m = _mesh()
        _result, reached = _drive(m, publisher, dict(_SMALL_CMD))
        assert len(reached) == 1, f"{publisher}() did not publish an in-cap command: {reached}"
        assert reached[0][1] <= cmd_bytes_cap()

    def test_sends_over_cap_error_wording_is_unchanged(self) -> None:
        # Operators and forks match on this text; the refactor moved where it
        # is built, not what it says.
        m = _mesh()
        result, _reached = _drive(m, "send", dict(_OVER_CAP_CMD))
        assert isinstance(result, dict)
        assert result["status"] == "error"
        encoded = len(
            json.dumps({"sender_id": "op", "turn_id": "x" * 32, "command": _OVER_CAP_CMD, "timestamp": 0.0}).encode(
                "utf-8"
            )
        )
        assert result["error"].startswith("command message is "), result["error"]
        assert result["error"].endswith("Shrink world_update/instruction or raise the cap on BOTH peers."), result[
            "error"
        ]
        # The reported size is the encoded message, not the bare command.
        assert abs(int(result["error"].split()[3]) - encoded) < 64, result["error"]

    def test_an_unreadable_cap_publishes_rather_than_refusing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The check is a diagnostic. If the config helper cannot be imported
        # there is no cap to compare against, so the payload goes out as it did
        # before the check existed rather than every command failing.
        m = _mesh()
        # A None entry makes the import raise ImportError from the import
        # machinery itself, which is what a missing optional module looks like.
        monkeypatch.setitem(sys.modules, "strands_robots.mesh._zenoh_config", None)
        assert m._cmd_topic_size_problem({"blob": "x" * 20000}) is None


class TestOneOwnerForTheSharedCap:
    """Structural: both publishers ask the helper, neither re-derives the cap."""

    @pytest.mark.parametrize("publisher", ["send", "broadcast"])
    def test_the_publisher_consults_the_shared_helper(self, publisher: str) -> None:
        src = textwrap.dedent(inspect.getsource(getattr(core.Mesh, publisher)))
        called = {
            ast.unparse(node.func).split(".")[-1] for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Call)
        }
        assert "_cmd_topic_size_problem" in called, (
            f"{publisher}() does not ask the shared cmd-size helper; a second inline check is how the "
            "two key expressions of one filter rule come to disagree about the cap."
        )

    @pytest.mark.parametrize("publisher", ["send", "broadcast"])
    def test_the_publisher_does_not_re_derive_the_cap(self, publisher: str) -> None:
        src = textwrap.dedent(inspect.getsource(getattr(core.Mesh, publisher)))
        assert "cmd_bytes_cap" not in src, (
            f"{publisher}() reads the cap itself instead of delegating; the helper owns it."
        )

    def test_the_helper_is_the_only_place_the_cap_is_read_on_this_path(self) -> None:
        src = textwrap.dedent(inspect.getsource(core.Mesh._cmd_topic_size_problem))
        assert "cmd_bytes_cap" in src
        assert "STRANDS_MESH_MAX_CMD_BYTES" in src, "the reason must name the knob that raises the cap"
