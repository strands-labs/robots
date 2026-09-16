"""Registry ``tool_frame``: a tool point for a model that ships no site.

``discover_ee_frame`` settles on a hand/wrist BODY when a model has no
tool-point site. For the SO-100 (``trs_so_arm100``, zero sites) that body is
``Wrist_Pitch_Roll``, ~16 cm short of the jaw tips, so every low target the
README's first robot was sent to was "reached" by the wrist while the fingers
hung in the air (0/3 in the v0.5.2 devx replay); the SO-101, whose model ships
a ``gripper`` site, landed 4/4 on the same prompt. The registry may now
declare the tool point the model lacks; the MuJoCo backend adds the site
before the attach, so discovery, ``get_robot_state`` and ``move_to`` follow it.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.tool_frame import (
    DEFAULT_TOOL_SITE,
    ToolFrame,
    registry_tool_frame,
    tool_frame_from_block,
)

GOOD = {"body": "Fixed_Jaw", "pos": [0.0, -0.0995, 0.001], "site": "tcp"}


class TestBlockShape:
    def test_a_well_formed_block_is_a_frame(self) -> None:
        frame, reason = tool_frame_from_block("so100", GOOD)
        assert reason is None
        assert frame == ToolFrame(body="Fixed_Jaw", pos=(0.0, -0.0995, 0.001), site="tcp")

    def test_site_defaults_to_tcp(self) -> None:
        frame, reason = tool_frame_from_block("x", {"body": "hand", "pos": [0, 0, 0.1]})
        assert reason is None and frame is not None
        assert frame.site == DEFAULT_TOOL_SITE == "tcp"

    def test_pos_ints_are_coerced_to_floats(self) -> None:
        frame, _ = tool_frame_from_block("x", {"body": "hand", "pos": [0, 0, 1]})
        assert frame is not None and frame.pos == (0.0, 0.0, 1.0)
        assert all(isinstance(v, float) for v in frame.pos)

    def test_a_block_that_cannot_be_rendered_is_still_answered(self) -> None:
        # The refusal IS the answer to an unusable block, so building it must not
        # raise - the package rule for refusal text (utils.refusal_repr).
        class Unprintable:
            def __repr__(self) -> str:
                raise RuntimeError("no repr for you")

        frame, reason = tool_frame_from_block("so100", Unprintable())
        assert frame is None
        assert reason is not None and "<unrepresentable Unprintable>" in reason

    @pytest.mark.parametrize(
        ("block", "fragment"),
        [
            ("Fixed_Jaw", "Expected {'body'"),
            ({"pos": [0, 0, 0]}, "'body' must be a non-empty body name"),
            ({"body": "", "pos": [0, 0, 0]}, "'body' must be a non-empty body name"),
            ({"body": "hand"}, "'pos' must be three finite numbers"),
            ({"body": "hand", "pos": [0, 0]}, "'pos' must be three finite numbers"),
            ({"body": "hand", "pos": [0, 0, float("nan")]}, "'pos' must be three finite numbers"),
            ({"body": "hand", "pos": [0, 0, True]}, "'pos' must be three finite numbers"),
            ({"body": "hand", "pos": ["0", 0, 0]}, "'pos' must be three finite numbers"),
            ({"body": "hand", "pos": [0, 0, 0], "site": ""}, "'site' must be a non-empty name without '/'"),
            ({"body": "hand", "pos": [0, 0, 0], "site": "arm/tcp"}, "'site' must be a non-empty name without '/'"),
            ({"body": "hand", "pos": [0, 0, 0], "offset": [0, 0, 0]}, "Unknown key(s) ['offset']"),
        ],
    )
    def test_a_malformed_block_names_the_entry_and_the_fault(self, block, fragment) -> None:
        frame, reason = tool_frame_from_block("so100", block)
        assert frame is None
        assert reason is not None
        assert reason.startswith("registry tool_frame for data_config 'so100' is malformed:")
        assert fragment in reason


class TestRegistryLookup:
    def test_no_data_config_means_no_frame(self) -> None:
        assert registry_tool_frame(None) == (None, None)
        assert registry_tool_frame("") == (None, None)

    def test_an_entry_without_the_block_means_no_frame(self) -> None:
        assert registry_tool_frame("so101", lookup=lambda _n: {"gripper": {}}) == (None, None)

    def test_an_unknown_entry_means_no_frame(self) -> None:
        assert registry_tool_frame("nope", lookup=lambda _n: None) == (None, None)

    def test_the_shipped_so100_entry_declares_a_jaw_tool_point(self) -> None:
        frame, reason = registry_tool_frame("so100")
        assert reason is None and frame is not None
        assert frame.body == "Fixed_Jaw" and frame.site == "tcp"

    def test_the_shipped_so101_entry_declares_none_its_model_has_a_site(self) -> None:
        assert registry_tool_frame("so101") == (None, None)

    def test_a_malformed_overlay_block_is_a_reason_not_a_fallback(self) -> None:
        frame, reason = registry_tool_frame("so100", lookup=lambda _n: {"tool_frame": {"body": "Fixed_Jaw"}})
        assert frame is None and reason is not None and "'pos'" in reason


def _mujoco():
    return pytest.importorskip("mujoco")


class TestSpecBuilderAddsTheSite:
    def _spec(self):
        mj = _mujoco()
        spec = mj.MjSpec()
        base = spec.worldbody.add_body(name="base")
        jaw = base.add_body(name="jaw")
        jaw.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0])
        return spec

    def test_the_site_lands_on_the_named_body_at_the_declared_pos(self) -> None:
        from strands_robots.simulation.mujoco.spec_builder import SpecBuilder

        spec = self._spec()
        SpecBuilder.add_tool_site(spec, "arm", ToolFrame(body="jaw", pos=(0.0, -0.1, 0.001), site="tcp"))
        model = spec.compile()
        mj = _mujoco()
        sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "tcp")
        assert sid >= 0
        assert mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.site_bodyid[sid])) == "jaw"
        assert model.site_pos[sid].tolist() == pytest.approx([0.0, -0.1, 0.001])

    def test_an_unknown_body_is_refused_naming_the_bodies_the_model_has(self) -> None:
        from strands_robots.simulation.mujoco.spec_builder import SpecBuilder
        from strands_robots.simulation.tool_frame import ToolFrameRefused

        spec = self._spec()
        with pytest.raises(ToolFrameRefused, match=r"names body 'Fixed_Jaw'.*Bodies in the model: \['base', 'jaw'\]"):
            SpecBuilder.add_tool_site(spec, "arm", ToolFrame(body="Fixed_Jaw", pos=(0, 0, 0)))
        assert [s.name for s in spec.sites] == []

    def test_a_site_name_the_model_already_uses_is_refused(self) -> None:
        from strands_robots.simulation.mujoco.spec_builder import SpecBuilder
        from strands_robots.simulation.tool_frame import ToolFrameRefused

        spec = self._spec()
        spec.body("jaw").add_site(name="tcp", pos=[0, 0, 0])
        with pytest.raises(ToolFrameRefused, match="already has a site of that name"):
            SpecBuilder.add_tool_site(spec, "arm", ToolFrame(body="jaw", pos=(0, 0, 0.05)))


# The three low targets the devx replay sent so100 to (its forward is -Y).
LOW_TARGETS = ([0.0, -0.25, 0.05], [0.05, -0.30, 0.08], [0.0, -0.20, 0.03])


def _sim(tool_name: str):
    _mujoco()
    pytest.importorskip("mink")
    from strands_robots.simulation.mujoco.simulation import Simulation

    sim = Simulation(tool_name=tool_name, mesh=False)
    sim.create_world()
    return sim


class TestOnTheMuJoCoBackend:
    def test_so100_gets_a_namespaced_tcp_site_that_discovery_and_state_follow(self) -> None:
        from strands_robots.simulation.ik import discover_ee_frame

        sim = _sim("tf_so100")
        try:
            assert sim.add_robot(name="arm", data_config="so100")["status"] == "success"
            model = sim._world._model
            mj = _mujoco()
            names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
            assert "arm/tcp" in names
            assert discover_ee_frame(model, "arm/") == ("arm/tcp", "site")
            text = sim.get_robot_state("arm")["content"][0]["text"]
            assert "end_effector (site 'arm/tcp', the frame move_to drives)" in text
        finally:
            sim.cleanup()

    def test_the_tool_point_sits_between_the_jaw_tips_not_at_the_wrist(self) -> None:
        import numpy as np

        sim = _sim("tf_so100_pos")
        try:
            sim.add_robot(name="arm", data_config="so100")
            mj = _mujoco()
            model, data = sim._world._model, sim._world._data
            mj.mj_forward(model, data)
            tcp = data.site_xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "arm/tcp")]
            wrist = data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "arm/Wrist_Pitch_Roll")]
            fixed = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "arm/Fixed_Jaw")
            moving = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "arm/Moving_Jaw")
            # Beyond both jaw body origins along the arm's forward (-Y), well past the wrist.
            assert tcp[1] < data.xpos[fixed][1] and tcp[1] < data.xpos[moving][1]
            assert 0.14 < float(np.linalg.norm(tcp - wrist)) < 0.18
        finally:
            sim.cleanup()

    def test_move_to_lands_the_replay_targets_that_the_wrist_frame_missed(self) -> None:
        sim = _sim("tf_so100_move")
        try:
            sim.add_robot(name="arm", data_config="so100")
            for target in LOW_TARGETS:
                result = sim.move_to(position=list(target), robot_name="arm")
                text = result["content"][0]["text"]
                assert result["status"] == "success", text
                assert "EE (site 'arm/tcp')" in text
        finally:
            sim.cleanup()

    def test_the_deprecated_name_as_registry_key_add_gets_the_site_too(self) -> None:
        # add_robot("so100") resolves the model through the deprecated
        # name-as-registry-key fallback; the tool point describes that model, so
        # it applies whichever argument named the entry.
        from strands_robots.simulation.ik import discover_ee_frame

        sim = _sim("tf_by_name")
        try:
            assert sim.add_robot(name="so100")["status"] == "success"
            assert discover_ee_frame(sim._world._model, "so100/") == ("so100/tcp", "site")
        finally:
            sim.cleanup()

    def test_a_model_with_its_own_site_is_left_alone(self) -> None:
        sim = _sim("tf_so101")
        try:
            sim.add_robot(name="arm", data_config="so101")
            mj = _mujoco()
            model = sim._world._model
            names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
            assert "arm/gripper" in names and "arm/tcp" not in names
        finally:
            sim.cleanup()

    def test_a_callers_own_model_keeps_its_frames_even_with_data_config_so100(self, tmp_path) -> None:
        # data_config= as the metadata source for a model that is NOT the
        # registry's: the so100 tool_frame names Fixed_Jaw, which this model
        # lacks - the declaration describes the registry's model, so it is not
        # applied here and the add is not refused.
        xml = """<mujoco model="inline_arm"><worldbody>
          <body name="base"><body name="hand" pos="0 0 0.2">
            <joint name="j1" type="hinge" axis="0 0 1"/><geom type="sphere" size="0.02"/>
          </body></body></worldbody>
          <actuator><position name="Jaw" joint="j1"/></actuator></mujoco>"""
        path = tmp_path / "inline_arm.xml"
        path.write_text(xml)
        sim = _sim("tf_own_model")
        try:
            result = sim.add_robot(name="arm", urdf_path=str(path), data_config="so100")
            assert result["status"] == "success", result["content"][0]["text"]
            mj = _mujoco()
            model = sim._world._model
            assert [mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)] == []
        finally:
            sim.cleanup()

    def test_a_malformed_overlay_block_refuses_add_robot_with_the_reason(self, monkeypatch) -> None:
        import strands_robots.simulation.mujoco.simulation as sim_mod

        monkeypatch.setattr(
            sim_mod,
            "registry_tool_frame",
            lambda key: (None, f"registry tool_frame for data_config '{key}' is malformed: 'nope'. Expected ..."),
        )
        sim = _sim("tf_bad")
        try:
            result = sim.add_robot(name="arm", data_config="so100")
            assert result["status"] == "error"
            assert "add_robot: registry tool_frame for data_config 'so100' is malformed" in result["content"][0]["text"]
            assert "arm" not in sim._world.robots
        finally:
            sim.cleanup()

    def test_a_body_the_model_lacks_refuses_add_robot_and_leaves_the_scene_clean(self, monkeypatch) -> None:
        import strands_robots.simulation.mujoco.simulation as sim_mod

        monkeypatch.setattr(
            sim_mod,
            "registry_tool_frame",
            lambda key: (ToolFrame(body="Not_A_Body", pos=(0, 0, 0)), None),
        )
        sim = _sim("tf_nobody")
        try:
            before = sim._world._backend_state["spec"].to_xml()
            result = sim.add_robot(name="arm", data_config="so100")
            assert result["status"] == "error"
            text = result["content"][0]["text"]
            assert "names body 'Not_A_Body', which the model does not have" in text
            assert "Fixed_Jaw" in text  # the bodies it does have are listed
            assert "arm" not in sim._world.robots
            # The check reads the robot's own spec before the attach, so the
            # scene's spec is not merely usable afterwards, it is untouched.
            assert sim._world._backend_state["spec"].to_xml() == before
            # And the next add succeeds.
            monkeypatch.setattr(sim_mod, "registry_tool_frame", lambda key: (None, None))
            assert sim.add_robot(name="arm2", data_config="so101")["status"] == "success"
        finally:
            sim.cleanup()
