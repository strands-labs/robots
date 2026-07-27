"""``add_object`` honors every ``color`` component or rejects the vector.

A MuJoCo geom stores its colour in a 4-component ``rgba`` row, so only two
component counts can be applied to it: an RGB triple (completed with the opaque
alpha the row defaults to) and a full RGBA quadruple. ``add_object`` used to
content-validate ``color`` without checking that count, which produced two
failures on the same parameter:

* An empty vector fell through the ``color or <default>`` coalescing and painted
  the mid-grey default under a ``status="success"`` result -- the caller asked
  for a colour and got the backend's.
* Any other partial vector (an RGB triple, a single component) reached MuJoCo's
  ``add_geom``, which refuses a non-4 ``rgba``, so the call died with a generic
  "spec recompile refused" while the actionable reason stayed in the log.

The same contract already governs the runtime mutator
(``set_geom_properties(color=...)``), which honors 3 or 4 components, so these
tests pin the creator to it -- including at the agent-tool dispatch boundary,
whose fixed-length table used to reject an RGB triple the method honors.
"""

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Counts a 4-component rgba row cannot be given without fabricating or
# discarding components the caller never mentioned.
UNHONORABLE_COLORS = [[], [1.0], [1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 9.0]]


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_add_object_color_count_sim", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _rgba(sim, name):
    gid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_GEOM, f"{name}_geom")
    assert gid >= 0, f"geom for {name!r} missing from the compiled model"
    return [round(float(c), 6) for c in sim._world._model.geom_rgba[gid]]


class TestAddObjectColorComponentCount:
    """The creator applies the colour it was given, or names why it cannot."""

    @pytest.mark.parametrize("color", UNHONORABLE_COLORS)
    def test_unhonorable_component_count_rejected(self, sim, color):
        """A colour the rgba row cannot hold is refused, and nothing is added.

        ``color=[]`` previously reported success with the default grey compiled,
        and the other counts previously died with "spec recompile refused".
        """
        result = sim.add_object("blob", shape="box", color=color)

        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "'color'" in text
        assert "3 or 4" in text
        assert "recompile" not in text
        assert "blob" not in sim._world.objects
        assert mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_GEOM, "blob_geom") < 0

    def test_rejection_leaves_the_name_free_and_the_scene_mutable(self, sim):
        """A refused colour does not consume the name or block later mutations."""
        assert sim.add_object("crate", shape="box", color=[0.2, 0.4])["status"] == "error"

        assert sim.add_object("crate", shape="box", color=[0.2, 0.4, 0.6])["status"] == "success"
        assert _rgba(sim, "crate") == [0.2, 0.4, 0.6, 1.0]

    def test_rgb_triple_completed_with_an_opaque_alpha(self, sim):
        """An RGB triple is honored -- alpha is the one component with a default."""
        result = sim.add_object("ball", shape="sphere", size=[0.1], color=[1.0, 0.0, 0.0])

        assert result["status"] == "success", result
        assert _rgba(sim, "ball") == [1.0, 0.0, 0.0, 1.0]

    def test_rgba_applied_verbatim_including_alpha(self, sim):
        """A 4-component colour keeps the alpha the caller chose."""
        assert sim.add_object("glass", shape="box", color=[0.1, 0.2, 0.3, 0.25])["status"] == "success"

        assert _rgba(sim, "glass") == [0.1, 0.2, 0.3, 0.25]

    @pytest.mark.parametrize("color", [np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.5])])
    def test_numpy_color_accepted(self, sim, color):
        """A NumPy colour (e.g. a ``geom_rgba`` row read back) is accepted.

        It used to reach ``color or <default>``, whose truth test on a multi
        element array raised ``ValueError`` past the tool-result contract.
        """
        assert sim.add_object("np_cube", shape="box", color=color)["status"] == "success"

        expected = [round(float(c), 6) for c in color]
        assert _rgba(sim, "np_cube") == (expected if len(expected) == 4 else [*expected, 1.0])

    def test_omitted_color_uses_the_documented_default(self, sim):
        """Only an omitted colour gets the mid-grey default, never a partial one."""
        assert sim.add_object("plain", shape="box")["status"] == "success"

        assert _rgba(sim, "plain") == [0.5, 0.5, 0.5, 1.0]


class TestColorContractSharedWithTheMutator:
    """The creator and the runtime mutator accept the same colour counts."""

    @pytest.mark.parametrize("color", [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]])
    def test_both_entry_points_honor_the_same_counts(self, sim, color):
        assert sim.add_object("shared", shape="box", color=color)["status"] == "success"
        expected = [*color, 1.0] if len(color) == 3 else color
        assert _rgba(sim, "shared") == expected

        assert sim.set_geom_properties(geom_name="shared", color=color)["status"] == "success"
        assert _rgba(sim, "shared") == expected

    @pytest.mark.parametrize("color", UNHONORABLE_COLORS)
    def test_both_entry_points_reject_the_same_counts(self, sim, color):
        assert sim.add_object("shared", shape="box", color=[0.5, 0.5, 0.5, 1.0])["status"] == "success"
        before = _rgba(sim, "shared")

        assert sim.add_object("other", shape="box", color=color)["status"] == "error"
        assert sim.set_geom_properties(geom_name="shared", color=color)["status"] == "error"
        assert _rgba(sim, "shared") == before


class TestColorComponentCountAtDispatch:
    """The agent-tool router forwards every count the method can honor."""

    def test_rgb_triple_reaches_the_method(self, sim):
        """The router used to reject an RGB triple that add_object honors."""
        result = sim._dispatch_action("add_object", {"name": "routed", "shape": "box", "color": [1.0, 0.0, 0.0]})

        assert result["status"] == "success", result
        assert _rgba(sim, "routed") == [1.0, 0.0, 0.0, 1.0]

    def test_rgb_triple_reaches_the_mutator(self, sim):
        assert sim.add_object("mutated", shape="box")["status"] == "success"

        result = sim._dispatch_action("set_geom_properties", {"geom_name": "mutated", "color": [0.0, 1.0, 0.0]})

        assert result["status"] == "success", result
        assert _rgba(sim, "mutated") == [0.0, 1.0, 0.0, 1.0]

    @pytest.mark.parametrize("color", [[1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 9.0]])
    def test_unhonorable_count_rejected_at_the_boundary(self, sim, color):
        result = sim._dispatch_action("add_object", {"name": "routed_bad", "shape": "box", "color": color})

        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "'color'" in text
        assert "3 or 4" in text
        assert "routed_bad" not in sim._world.objects

    def test_fixed_length_vector_params_still_report_their_single_count(self, sim):
        """Widening the table must not blur a param with one honorable count."""
        result = sim._dispatch_action("add_object", {"name": "bad_pos", "shape": "box", "position": [0.0, 1.0]})

        assert result["status"] == "error", result
        assert "must be a list of 3 numbers, got 2." in result["content"][0]["text"]
