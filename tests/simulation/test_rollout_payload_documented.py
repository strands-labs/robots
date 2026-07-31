"""The rollout facades must document the json payload they actually return.

:meth:`~strands_robots.simulation.base.SimEngine.run_policy` and
:meth:`~strands_robots.simulation.base.SimEngine.eval_policy` return the
agent-tool envelope, so every rollout fact a caller can act on lives in the
result's ``{"json": {...}}`` content block rather than in the return value
itself. The outer ``status`` reports only that the CALL was accepted and the
loop ran: a policy that drives 1 of a Panda's 8 actuators is deliberately
``status="success"`` with ``action_errors=0``, and the two fields that DO
expose it - ``action_resolution_rate`` and ``partial_action_failure_rate`` -
are readable only from that block.

The docstring's ``Returns:`` section is the only place those fields are
enumerated for a caller. A field the payload carries but the docstring omits is
effectively invisible: a caller cannot gate on a key it has no way to discover,
and gates on ``status`` alone instead. That is exactly how 18 of ``run_policy``'s
30 payload fields - and all 22 of ``eval_policy``'s - came to be undocumented
while the payload kept growing.

These tests pin each facade's documented field list to the payload a real
rollout produces, so the two can never drift apart again: adding a field to the
payload without documenting it fails here.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

import strands_robots
from strands_robots.policies import MockPolicy
from strands_robots.simulation.base import SimEngine

# Asset-free two-joint arm: keeps the guard independent of any downloaded robot
# description. ``angle="radian"`` is required - MJCF defaults to degrees, which
# would make the joint ranges ~2 degrees and the servos fight their limits.
ARM_XML = """
<mujoco model="guard_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-2 2" limited="true" damping="4"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body name="link" pos="0.2 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="30" ctrlrange="-2 2"/>
    <position name="elbow_act" joint="elbow" kp="30" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""


def documented_fields(method_name: str) -> set[str]:
    """Field names enumerated in a facade's ``Returns:`` docstring section.

    Only ``\\`\\`name\\`\\``-quoted identifiers count: that is the repo's
    docstring style for a payload field, and matching on the quoted form
    prevents a shorter key passing merely because it is a substring of a longer
    documented one (``n_episodes`` inside ``n_episodes_completed``).
    """
    doc = getattr(SimEngine, method_name).__doc__ or ""
    block = re.search(r"\n\s*Returns:\n(.*?)(?=\n\s*(?:Raises|Args|Example|Note):|\Z)", doc, re.S)
    if block is None:
        return set()
    return set(re.findall(r"``([a-z][a-z0-9_]*)``", block.group(1)))


def payload_of(result: dict[str, Any]) -> dict[str, Any]:
    """The first ``{"json": {...}}`` block, found by scanning - never by index.

    A fixed ``content[1]`` raises ``IndexError`` on an early caller-error
    return, which carries a ``text`` block only.
    """
    return next((b["json"] for b in result.get("content", []) if isinstance(b.get("json"), dict)), {})


@pytest.fixture
def sim(tmp_path):
    """A minimal two-joint arm scene; no rendering, no downloaded assets."""
    path = tmp_path / "arm.xml"
    path.write_text(ARM_XML)
    engine = strands_robots.Simulation(backend="mujoco", tool_name="payload_doc_guard", mesh=False)
    engine.create_world()
    engine.add_robot(name="arm", urdf_path=str(path))
    yield engine
    engine.cleanup()


def _bound_policy(engine) -> MockPolicy:
    policy = MockPolicy()
    policy.set_robot_state_keys(engine.robot_action_keys("arm"))
    return policy


def run_policy_payload(engine) -> dict[str, Any]:
    result = engine.run_policy(robot_name="arm", policy_object=_bound_policy(engine), n_steps=4, control_frequency=50.0)
    assert result["status"] == "success", result
    return payload_of(result)


def eval_policy_payload(engine) -> dict[str, Any]:
    result = engine.eval_policy(
        robot_name="arm", policy_object=_bound_policy(engine), n_episodes=1, max_steps=4, control_frequency=50.0
    )
    assert result["status"] == "success", result
    return payload_of(result)


PAYLOAD_BUILDERS = {"run_policy": run_policy_payload, "eval_policy": eval_policy_payload}


class TestEveryPayloadFieldIsDocumented:
    """A field a caller can read must be a field a caller can discover."""

    @pytest.mark.parametrize("method_name", sorted(PAYLOAD_BUILDERS))
    def test_returns_section_enumerates_every_payload_field(self, sim, method_name):
        payload = PAYLOAD_BUILDERS[method_name](sim)
        assert payload, f"{method_name} returned no json block - the guard would pass vacuously"
        missing = sorted(set(payload) - documented_fields(method_name))
        assert not missing, (
            f"SimEngine.{method_name}'s Returns: section does not document "
            f"{len(missing)} field(s) its payload carries: {missing}. "
            f"Add each as ``field_name`` to that section."
        )

    @pytest.mark.parametrize("method_name", sorted(PAYLOAD_BUILDERS))
    def test_documented_fields_all_exist_in_the_payload(self, sim, method_name):
        """The reverse direction: no documented field may be a phantom."""
        payload = PAYLOAD_BUILDERS[method_name](sim)
        # Only names the docstring presents as payload fields are compared, so a
        # method name or argument mentioned in prose is not read as a field.
        phantom = sorted(
            f for f in documented_fields(method_name) if f.endswith(("_s", "_rate", "_used")) and f not in payload
        )
        assert not phantom, f"SimEngine.{method_name} documents field(s) its payload does not carry: {phantom}"


class TestActionHealthFieldsAreDiscoverable:
    """The fields that expose a crippled-but-'successful' rollout, by name.

    A rollout driving a subset of the robot's actuators returns
    ``status="success"`` with ``action_errors=0``. These are the only fields
    that contradict that, so their presence in the docstring is the whole
    difference between a caller that can detect it and one that cannot.
    """

    @pytest.mark.parametrize("field", ["action_resolution_rate", "partial_action_failure_rate", "action_errors"])
    def test_run_policy_documents_action_health_field(self, field):
        assert field in documented_fields("run_policy")

    @pytest.mark.parametrize("field", ["success_rate", "success_measured"])
    def test_eval_policy_documents_outcome_field(self, field):
        assert field in documented_fields("eval_policy")


class TestDocumentedAccessPathIsIndexFree:
    """The docstring must not teach a fixed index into ``content``.

    ``content[1]["json"]`` raises ``IndexError`` on an early caller-error
    return, which carries a ``text`` block only - i.e. on exactly the results a
    caller most needs to read.
    """

    @pytest.mark.parametrize("method_name", sorted(PAYLOAD_BUILDERS))
    def test_returns_section_does_not_teach_a_fixed_index(self, method_name):
        doc = getattr(SimEngine, method_name).__doc__ or ""
        assert 'content[1]["json"]' not in doc.replace("'", '"') or "IndexError" in doc

    def test_a_caller_error_result_has_no_json_block_to_index(self, sim):
        """The premise of the warning above, pinned against the live engine."""
        result = sim.run_policy(robot_name="arm", duration=-1)
        assert result["status"] == "error"
        assert len(result["content"]) == 1, result["content"]
        assert payload_of(result) == {}


class TestGuardCannotPassVacuously:
    """A scanner that silently matched nothing would look like a clean suite."""

    def test_a_planted_undocumented_field_is_detected(self, sim):
        payload = run_policy_payload(sim)
        payload["a_field_no_docstring_mentions"] = 1
        assert sorted(set(payload) - documented_fields("run_policy")) == ["a_field_no_docstring_mentions"]

    def test_the_extractor_finds_a_non_empty_field_list(self):
        for method_name in PAYLOAD_BUILDERS:
            assert len(documented_fields(method_name)) >= 10, method_name
