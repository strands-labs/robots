"""add_robot surfaces an actionable error for an unknown/unresolvable model.

Contract: when a caller names a robot that resolves to no model (a mistyped
registry key, or an instance label given without a model source), the engine
returns the same actionable "no model found (did you mean ...?) / list_urdfs /
pass data_config= or urdf_path=" error that the ``data_config`` exit and the
top-level ``Robot()`` factory give - not a dead-end "Either urdf_path or
data_config is required" that never names the robot nor points at discovery.

The bare "supply a model source" message is preserved for the genuine no-name
case (``add_robot()`` with nothing to resolve), and the deprecated positional
name-as-registry-key short form still resolves a VALID name.
"""

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture
def world():
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    sim = MuJoCoSimEngine(tool_name="test_add_robot_unknown_model")
    sim.create_world()
    yield sim
    sim.cleanup()


def _msg(result):
    assert result["status"] == "error", f"expected error, got: {result}"
    return result["content"][0]["text"]


class TestUnknownModelMessage:
    def test_positional_typo_is_actionable(self, world):
        """A mistyped positional robot name names the robot + offers discovery."""
        msg = _msg(world.add_robot("panda_typo"))
        assert "panda_typo" in msg, msg
        assert "list_urdfs" in msg, msg
        # dead-end message must NOT be what a named typo gets
        assert "Either urdf_path or data_config is required" not in msg, msg

    def test_typo_offers_close_match_suggestion(self, world):
        """A near-miss of a real robot suggests the correct name (difflib)."""
        msg = _msg(world.add_robot("panda_typo"))
        assert "panda" in msg, msg  # 'Did you mean: panda, ...'

    def test_instance_label_without_model_points_at_both_options(self, world):
        """A name given without a model source explains BOTH interpretations:
        pick a registered model (list_urdfs) OR supply data_config=/urdf_path=."""
        msg = _msg(world.add_robot(name="myarm"))
        assert "myarm" in msg, msg
        assert "list_urdfs" in msg, msg
        assert "data_config" in msg and "urdf_path" in msg, msg

    def test_no_args_keeps_generic_model_source_message(self, world):
        """No caller-provided name -> the generic 'supply a model source' error
        is preserved (the actionable unknown-robot message is name-specific)."""
        msg = _msg(world.add_robot())
        assert msg == "Either urdf_path or data_config is required.", msg

    def test_deprecated_positional_valid_name_still_resolves(self, world):
        """The deprecated name-as-registry-key short form still resolves a VALID
        name past the unknown-model gate (regression guard on the fallback)."""
        result = world.add_robot("so100")
        txt = result["content"][0]["text"] if result.get("content") else ""
        assert "No model found" not in txt, txt
