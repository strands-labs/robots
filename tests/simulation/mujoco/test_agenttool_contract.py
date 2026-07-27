"""T1/T13: AgentTool router contract - unknown kwargs rejected, required args friendly,
vector dims validated, tool_spec matches method signatures."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation(tool_name="contract_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


# Minimal 2-joint arm with a named camera, used to exercise the per-robot
# scoping branch of get_features (namespaced joint/actuator/camera pools).
_SCOPING_ARM_XML = """<mujoco model="scoping_arm">
  <worldbody>
    <camera name="wrist" pos="1.5 0 1" xyaxes="0 1 0 -0.5 0 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2"/>
        <joint name="lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="50"/>
    <position name="lift_act" joint="lift" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def arm_xml(tmp_path):
    """Write the scoping-arm MJCF to a temp file and return its path."""
    path = tmp_path / "scoping_arm.xml"
    path.write_text(_SCOPING_ARM_XML)
    return str(path)


class TestRouterRejectsUnknownKwargs:
    """T1 DoD: Unknown top-level params must be rejected with a clear message."""

    def test_unknown_kwarg_on_set_gravity(self, sim):
        result = sim._dispatch_action("set_gravity", {"gravity": [0, 0, -9.81], "bogus_param": 42})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Unknown parameter 'bogus_param'" in text
        assert "set_gravity" in text
        assert "Valid:" in text

    def test_unknown_kwarg_on_step(self, sim):
        result = sim._dispatch_action("step", {"n_steps": 5, "num_steps": 10})
        assert result["status"] == "error"
        assert "Unknown parameter 'num_steps'" in result["content"][0]["text"]

    def test_unknown_kwarg_on_reset(self, sim):
        result = sim._dispatch_action("reset", {"hard_reset": True})
        assert result["status"] == "error"
        assert "Unknown parameter 'hard_reset'" in result["content"][0]["text"]


class TestRouterRequiredArgError:
    """T1 DoD: Missing required params produce a friendly error (no Python TypeError)."""

    def test_missing_required_arg_on_add_object(self, sim):
        # add_object requires `name`. Default for shape is `box` but `name` has no default.
        result = sim._dispatch_action("add_object", {"shape": "box"})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "requires parameter 'name'" in text
        assert "add_object" in text

    def test_missing_required_arg_on_stop_policy(self, sim):
        # stop_policy has robot_name default="" so it's not technically required;
        # but apply_force requires body_name.
        result = sim._dispatch_action("apply_force", {"force": [0, 0, 1]})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "requires parameter 'body_name'" in text


class TestRouterValidatesVectorDims:
    """T1 DoD: Vector params with wrong length rejected before reaching MuJoCo."""

    def test_gravity_wrong_length_rejected(self, sim):
        result = sim._dispatch_action("set_gravity", {"gravity": [0, 0]})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "'gravity'" in text and "3" in text and "2" in text

    def test_position_wrong_length_rejected(self, sim):
        result = sim._dispatch_action(
            "add_object",
            {"name": "box1", "shape": "box", "position": [0, 0]},
        )
        assert result["status"] == "error"
        assert "'position'" in result["content"][0]["text"]

    def test_orientation_wrong_length_rejected(self, sim):
        # orientation is a quaternion (4)
        result = sim._dispatch_action(
            "add_object",
            {"name": "box1", "shape": "box", "orientation": [1, 0, 0]},
        )
        assert result["status"] == "error"
        assert "'orientation'" in result["content"][0]["text"]

    def test_color_unhonorable_length_rejected(self, sim):
        # color is rgb (3) or rgba (4); two components cannot be applied to the
        # geom's rgba row without inventing the rest.
        result = sim._dispatch_action(
            "add_object",
            {"name": "box1", "shape": "box", "color": [1, 0]},
        )
        assert result["status"] == "error"
        assert "'color'" in result["content"][0]["text"]

    def test_color_rgb_triple_forwarded(self, sim):
        # The router used to require exactly 4 components, rejecting an RGB
        # triple that add_object honors (alpha defaults to opaque).
        result = sim._dispatch_action(
            "add_object",
            {"name": "box_rgb", "shape": "box", "color": [1, 0, 0]},
        )
        assert result["status"] == "success", result

    def test_non_numeric_vector_component_rejected(self, sim):
        result = sim._dispatch_action("set_gravity", {"gravity": [0, 0, "low"]})
        assert result["status"] == "error"
        assert "numeric" in result["content"][0]["text"].lower()

    def test_non_list_vector_rejected(self, sim):
        result = sim._dispatch_action("set_gravity", {"gravity": 9.81})
        assert result["status"] == "error"
        assert "'gravity'" in result["content"][0]["text"]

    def test_none_vector_param_skips_validation(self, sim):
        # An explicit ``None`` for an optional vector param means "use the
        # default", so the router must skip dimension/dtype validation and let
        # the value flow through rather than rejecting it as malformed.
        result = sim._dispatch_action(
            "add_object",
            {"name": "box_none_orient", "shape": "box", "orientation": None},
        )
        assert result["status"] == "success"


class TestRouterNameRobotNameAlias:
    """The router aliases ``name`` and ``robot_name`` in both directions so an
    agent can use either spelling regardless of which one the target method
    declares. Without this, a caller guessing the wrong name gets a spurious
    "requires parameter" error instead of reaching the method.
    """

    def test_robot_name_alias_feeds_a_name_param(self, sim):
        # remove_object's signature uses ``name``; passing ``robot_name`` must
        # resolve to it. Proof: we reach the method (which reports the object
        # is missing) instead of failing validation with "requires parameter".
        result = sim._dispatch_action("remove_object", {"robot_name": "ghost"})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "ghost" in text
        assert "requires parameter" not in text

    def test_name_alias_feeds_a_robot_name_param(self, sim):
        # stop_policy's signature uses ``robot_name``; passing ``name`` must
        # resolve to it. Proof: the method reports the named robot is unknown,
        # rather than treating robot_name as empty (its default).
        result = sim._dispatch_action("stop_policy", {"name": "ghost"})
        assert result["status"] == "error"
        assert "ghost" in result["content"][0]["text"]


class TestRouterKwargsPassthrough:
    """Methods declaring **kwargs receive the residual keys and own the check.

    The router's unknown-parameter guard is skipped for those methods, so it
    must hand the residual keys over rather than drop them: dropping would leave
    the input neither validated nor used, which is how ``randomize`` /
    ``set_obs_noise`` came to report a misspelled parameter as a successful
    no-op (see test_domain_randomization_rejects_unknown_params.py). A
    forwarding sink passes the keys to its callee; a discarding sink rejects
    them. Verified directly against the validator with a signature-only stub.
    """

    def test_router_forwards_residual_keys_to_var_keyword_methods(self, sim):
        import inspect

        def fake(self, name, **kwargs):  # noqa: ANN001, ANN002, ANN202
            """Signature-only stub; never called, its body is irrelevant."""

        sig = inspect.signature(fake)
        kwargs, err = sim._validate_and_build_kwargs("fake", "fake", sig, {"name": "x", "future_flag": True})
        assert err is None
        assert kwargs == {"name": "x", "future_flag": True}

    def test_router_still_rejects_unknown_keys_without_var_keyword(self, sim):
        import inspect

        def fake(self, name):  # noqa: ANN001, ANN202
            """Signature-only stub; never called, its body is irrelevant."""

        sig = inspect.signature(fake)
        kwargs, err = sim._validate_and_build_kwargs("fake", "fake", sig, {"name": "x", "future_flag": True})
        assert kwargs is None
        assert err is not None and "future_flag" in err["content"][0]["text"]


class TestToolSpecMethodParity:
    """T13 DoD: every enum action in tool_spec.json has a matching method whose
    signature matches declared top-level params."""

    # Params in tool_spec.json that are intentionally not consumed by every method
    # (they are cross-cutting or action-conditional).
    SPEC_ONLY_ALLOWED = {
        # action is the dispatch key itself
        "action",
        # video composite params - folded into `video` by the router
        "output_path",
        "fps",
        # name/robot_name are aliased bi-directionally
        "robot_name",
        "name",
        # global knobs sometimes listed at top level for LLM convenience
    }

    def test_every_action_maps_to_a_method(self, sim):
        import strands_robots.simulation.mujoco as _mj_mod

        spec_path = Path(_mj_mod.__file__).parent / "tool_spec.json"
        spec = json.loads(spec_path.read_text())
        actions = spec["properties"]["action"]["enum"]

        missing = []
        for action in actions:
            method_name = sim._ACTION_ALIASES.get(action, action)
            if not hasattr(sim, method_name):
                missing.append(action)
        assert not missing, f"Actions without a method: {missing}"

    def test_no_method_has_silently_unused_param(self, sim):
        """Known legacy drifts that the router USED to silently drop are now
        either implemented or flagged by the router. This test enumerates
        the pre-T1 drift cases as a regression ward."""
        # Before T1: step(num_steps), run_policy(n_steps wrong), etc. silently dropped.
        # After T1: all of these rejected. Verify a sampling.
        drift_cases = [
            ("step", {"num_steps": 5}),  # should be `n_steps`
            ("forward_kinematics", {"some_ghost_param": 1}),
            ("get_features", {"unknown_filter": "a"}),
        ]
        for action, bad_kwargs in drift_cases:
            result = sim._dispatch_action(action, bad_kwargs)
            # Router must reject; must NOT silently succeed with default values.
            assert result["status"] == "error", f"{action} silently accepted {bad_kwargs}"

    def test_recompile_world_dead_helper_is_gone(self, sim):
        """`_recompile_world` was an orphaned "nuke and pave" rebuild helper: no
        caller invoked it (the scene-load path rebuilds through
        ``scene_ops._recompile_preserving_state``), and it was never reachable
        through the action dispatcher either -- ``_dispatch_action`` rejects any
        leading-underscore action and no ``_ACTION_ALIASES`` entry maps to it.
        Pin its removal so an unreachable, uncoverable branch cannot creep back."""
        assert not hasattr(type(sim), "_recompile_world")
        for action in ("recompile_world", "_recompile_world"):
            result = sim._dispatch_action(action, {})
            assert result["status"] == "error"
            assert "Unknown action" in result["content"][0]["text"]


class TestUnifiedNoWorldMessage:
    """T14: Every action must use the same 'No world.' message when no world exists."""

    @pytest.fixture
    def fresh_sim(self):
        """A sim with NO world."""
        s = Simulation(tool_name="no_world_test", mesh=False)
        yield s
        s.cleanup()

    def _assert_standard_no_world_error(self, result, action):
        assert result["status"] == "error", f"{action} should error when no world"
        text = result["content"][0]["text"]
        assert "No world" in text, f"{action} error text lacks 'No world': {text}"

    def test_step_no_world(self, fresh_sim):
        self._assert_standard_no_world_error(fresh_sim._dispatch_action("step", {"n_steps": 1}), "step")

    def test_reset_no_world(self, fresh_sim):
        self._assert_standard_no_world_error(fresh_sim._dispatch_action("reset", {}), "reset")

    def test_set_gravity_no_world(self, fresh_sim):
        self._assert_standard_no_world_error(
            fresh_sim._dispatch_action("set_gravity", {"gravity": [0, 0, -1]}),
            "set_gravity",
        )

    def test_render_no_world(self, fresh_sim):
        # render returns error cleanly when no world, not a crash.
        result = fresh_sim._dispatch_action("render", {})
        assert result["status"] == "error"
        # render uses the unified message now:
        assert "No world" in result["content"][0]["text"]

    def test_get_state_no_world(self, fresh_sim):
        self._assert_standard_no_world_error(fresh_sim._dispatch_action("get_state", {}), "get_state")


class TestUnifiedNotFoundMessages:
    """T15: Unknown-name errors use the consistent '<Kind> X not found.' shape."""

    def test_robot_not_found(self, sim):
        result = sim._dispatch_action("get_robot_state", {"robot_name": "ghost_bot"})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Robot 'ghost_bot' not found" in text

    def test_object_not_found(self, sim):
        result = sim._dispatch_action("move_object", {"name": "ghost_box", "position": [0, 0, 0]})
        assert result["status"] == "error"
        assert "Object 'ghost_box' not found" in result["content"][0]["text"]

    def test_body_not_found(self, sim):
        result = sim._dispatch_action("apply_force", {"body_name": "ghost_body", "force": [0, 0, 1]})
        assert result["status"] == "error"
        assert "Body 'ghost_body' not found" in result["content"][0]["text"]

    def test_sensor_not_found(self, sim):
        result = sim._dispatch_action("get_sensor_data", {"sensor_name": "ghost_sensor"})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        # T45 is about distinguishing "no sensors" vs "not found"; at minimum the
        # current behaviour must mention the sensor name clearly.
        assert "ghost_sensor" in text


class TestIdempotentStopFamily:
    """T16: stop_recording, stop_cameras_recording, stop_policy and close_viewer
    can be called unconditionally - when already stopped they succeed with a
    distinguishable 'Was not ...' message."""

    def test_stop_recording_twice_is_idempotent(self, sim):
        r1 = sim.stop_recording()
        assert r1["status"] == "success"
        r2 = sim.stop_recording()
        assert r2["status"] == "success"
        assert "Was not recording" in r2["content"][0]["text"]

    def test_stop_cameras_recording_twice_is_idempotent(self, sim):
        r1 = sim.stop_cameras_recording()
        assert r1["status"] == "success"
        r2 = sim.stop_cameras_recording()
        assert r2["status"] == "success"

    def test_close_viewer_twice_is_idempotent(self, sim):
        # close_viewer was already idempotent - pin it with a regression test.
        assert sim.close_viewer()["status"] == "success"
        assert sim.close_viewer()["status"] == "success"


class TestStopPolicyContract:
    """T16 + T24: stop_policy requires a robot_name; is idempotent per robot."""

    def test_stop_policy_empty_robot_name_friendly_error(self, sim):
        r = sim._dispatch_action("stop_policy", {})
        assert r["status"] == "error"
        assert "requires" in r["content"][0]["text"].lower() and "robot_name" in r["content"][0]["text"]

    def test_stop_policy_unknown_robot_errors(self, sim):
        r = sim._dispatch_action("stop_policy", {"robot_name": "ghost_bot"})
        assert r["status"] == "error"
        assert "Robot 'ghost_bot' not found" in r["content"][0]["text"]


class TestForwardPassBeforeReads:
    """T18/T19: get_mass_matrix, get_contacts run mj_forward first so values
    are valid immediately after a reset / add_robot / load_state, not just
    after a full mj_step."""

    def test_get_mass_matrix_after_reset_is_valid(self, sim):
        sim.reset()
        r = sim._dispatch_action("get_mass_matrix", {})
        assert r["status"] == "success"
        # Empty scene: nv==0 so rank==0 and cond==inf are acceptable; the
        # important bit is we didn't return NaN / raise.
        payload = r["content"][-1].get("json", {}) if isinstance(r["content"][-1], dict) else {}
        assert "shape" in payload

    def test_get_contacts_at_t0_no_phantom_penetrations(self, sim):
        # Empty world has no contacts; running this at t=0 must succeed
        # and return an empty list (T19 used to surface stale/uninit data).
        sim.reset()
        r = sim._dispatch_action("get_contacts", {})
        assert r["status"] == "success"
        payload = r["content"][-1]["json"] if isinstance(r["content"][-1], dict) else {}
        contacts = payload.get("contacts", [])
        # An empty world has no contacts. If the fix isn't applied and stale
        # data surfaces, contacts may contain garbage names/distances. Assert
        # either empty or all distances > -1mm (no phantom deep penetrations).
        for c in contacts:
            assert c["dist"] > -0.001, f"phantom penetration: {c}"


class TestRenderDimValidation:
    """T20: non-positive width/height rejected; oversized dims get plain-English
    message instead of raw MuJoCo framebuffer error."""

    def test_zero_width_rejected(self, sim):
        r = sim._dispatch_action("render", {"width": 0, "height": 120})
        assert r["status"] == "error"
        assert "width and height must be > 0" in r["content"][0]["text"]

    def test_negative_height_rejected(self, sim):
        r = sim._dispatch_action("render", {"width": 160, "height": -10})
        assert r["status"] == "error"
        assert "must be > 0" in r["content"][0]["text"]

    def test_oversize_dim_message_is_friendly(self, sim):
        # Request 8000x8000 - well above any sane offscreen framebuffer cap.
        r = sim._dispatch_action("render", {"width": 8000, "height": 8000})
        assert r["status"] == "error"
        text = r["content"][0]["text"]
        assert "exceeds" in text
        assert "framebuffer" in text
        assert "offwidth" in text  # points at the fix


class TestRenderDepthSurfaces:
    """T21: render_depth mac warning surfaces in the response text when the
    driver lacks ARB_clip_control. Skipped when the warning isn't triggered
    (Linux / modern macOS GPUs may or may not hit it)."""

    def test_render_depth_returns_well_formed_response(self, sim):
        # Just check render_depth runs cleanly; the T21-specific warning
        # only fires on macOS without ARB_clip_control so we only assert
        # presence-of-warning when _depth_warn_text is set.
        r = sim._dispatch_action("render_depth", {})
        # Some headless envs don't have GL: we only care the response shape
        # is valid either way.
        assert r["status"] in ("success", "error")
        if r["status"] == "success":
            text = r["content"][0]["text"]
            # If a warning was captured, it must be on the response.
            warn_cached = getattr(sim, "_depth_warn_text", "")
            if warn_cached:
                assert warn_cached in text


class TestFeatureFilters:
    """T32 / T33: forward_kinematics + get_features honor per-entity filters."""

    def test_forward_kinematics_body_name_filters(self, sim):
        # Empty world: world body exists but any custom name is absent.
        r = sim._dispatch_action("forward_kinematics", {"body_name": "ghost_body"})
        assert r["status"] == "error"
        assert "Body 'ghost_body' not found" in r["content"][0]["text"]

    def test_forward_kinematics_no_filter_returns_all(self, sim):
        r = sim._dispatch_action("forward_kinematics", {})
        assert r["status"] == "success"
        payload = r["content"][-1]["json"] if isinstance(r["content"][-1], dict) else {}
        assert "bodies" in payload

    def test_get_features_unknown_robot_errors(self, sim):
        r = sim._dispatch_action("get_features", {"robot_name": "ghost_bot"})
        assert r["status"] == "error"
        assert "Robot 'ghost_bot' not found" in r["content"][0]["text"]

    def test_get_features_no_filter_returns_all(self, sim):
        r = sim._dispatch_action("get_features", {})
        assert r["status"] == "success"

    def test_get_features_scoped_to_robot_namespaces_pools(self, sim, arm_xml):
        """With multiple robots loaded, get_features(robot_name=X) restricts the
        actuator / camera pools to X's namespaced MuJoCo names and the robots
        map to just X - it does not leak the sibling robot's entities."""
        assert sim.add_robot("arm1", urdf_path=arm_xml, position=[0, 0, 0])["status"] == "success"
        assert sim.add_robot("arm2", urdf_path=arm_xml, position=[0.5, 0, 0])["status"] == "success"

        r = sim._dispatch_action("get_features", {"robot_name": "arm2"})
        assert r["status"] == "success"
        features = r["content"][-1]["json"]["features"]

        # robots map is filtered to just the requested robot.
        assert list(features["robots"].keys()) == ["arm2"]
        info = features["robots"]["arm2"]
        assert info["n_joints"] == 2
        assert info["n_actuators"] == 2

        # Actuator + camera pools are namespace-scoped to arm2 only - no arm1 leak.
        assert features["actuator_names"] == ["arm2/pan_act", "arm2/lift_act"]
        assert all(n.startswith("arm2/") for n in features["actuator_names"])
        assert all(n.startswith("arm2/") for n in features["camera_names"])
        assert "arm2/wrist" in features["camera_names"]
        assert not any(n.startswith("arm1/") for n in features["actuator_names"])

        # The robot's own (un-namespaced) joint names come straight from the robot.
        assert features["joint_names"] == ["pan", "lift"]

    def test_get_features_unscoped_lists_all_robots(self, sim, arm_xml):
        """Without a robot_name filter, get_features reports every robot and the
        full (cross-robot) actuator pool."""
        sim.add_robot("arm1", urdf_path=arm_xml, position=[0, 0, 0])
        sim.add_robot("arm2", urdf_path=arm_xml, position=[0.5, 0, 0])

        r = sim._dispatch_action("get_features", {})
        assert r["status"] == "success"
        features = r["content"][-1]["json"]["features"]
        assert set(features["robots"].keys()) == {"arm1", "arm2"}
        # Full pool spans both robots' namespaced actuators.
        assert "arm1/pan_act" in features["actuator_names"]
        assert "arm2/pan_act" in features["actuator_names"]


class TestRegisterUrdfValidation:
    """T35 / T42: register_urdf validates path + router covers no-args."""

    def test_register_urdf_no_args_friendly_error(self, sim):
        r = sim._dispatch_action("register_urdf", {})
        assert r["status"] == "error"
        assert "requires parameter" in r["content"][0]["text"]

    def test_register_urdf_missing_file_errors(self, sim):
        r = sim._dispatch_action(
            "register_urdf",
            {"data_config": "my_bot", "urdf_path": "/nonexistent/nope.urdf"},
        )
        assert r["status"] == "error"
        assert "file not found" in r["content"][0]["text"].lower()

    def test_register_urdf_empty_path_errors(self, sim):
        r = sim._dispatch_action("register_urdf", {"data_config": "my_bot", "urdf_path": ""})
        assert r["status"] == "error"
        # Router handles empty string as missing? No - it's a truthy string
        # in the presence test. So we hit our explicit empty guard.
        assert "non-empty" in r["content"][0]["text"] or "requires parameter" in r["content"][0]["text"]

    def test_register_urdf_directory_path_errors(self, sim, tmp_path):
        # A path that exists but is a directory must be rejected with a
        # distinct "not a file" message, not the generic "file not found"
        # (the latter would mislead an agent into thinking the path is wrong
        # when it is actually pointing at the wrong kind of node).
        r = sim._dispatch_action(
            "register_urdf",
            {"data_config": "my_bot", "urdf_path": str(tmp_path)},
        )
        assert r["status"] == "error"
        assert "not a file" in r["content"][0]["text"].lower()

    def test_register_urdf_unreadable_file_errors(self, sim, tmp_path):
        # An existing regular file that cannot be opened (permission denied)
        # is caught up front with an actionable "cannot read" message rather
        # than deferring to a cryptic MuJoCo load error later.
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root bypasses filesystem permission checks")
        urdf = tmp_path / "locked.urdf"
        urdf.write_text("<mujoco/>")
        urdf.chmod(0o000)
        if os.access(urdf, os.R_OK):
            pytest.skip("filesystem does not enforce read permissions here")
        try:
            r = sim._dispatch_action(
                "register_urdf",
                {"data_config": "my_bot", "urdf_path": str(urdf)},
            )
        finally:
            urdf.chmod(0o644)
        assert r["status"] == "error"
        assert "cannot read" in r["content"][0]["text"].lower()


class TestDuplicateCameraName:
    """T30 / T41: add_camera rejects duplicate names instead of silently
    overwriting the registry entry while leaving the XML unchanged."""

    def test_duplicate_camera_rejected(self, sim):
        r1 = sim._dispatch_action(
            "add_camera",
            {"name": "dupe", "position": [0.5, 0.5, 0.5], "target": [0, 0, 0]},
        )
        assert r1["status"] == "success", r1
        r2 = sim._dispatch_action(
            "add_camera",
            {"name": "dupe", "position": [1, 0, 0], "target": [0, 0, 0]},
        )
        assert r2["status"] == "error"
        assert "already exists" in r2["content"][0]["text"]


class TestPlaneAutoStatic:
    """T29: add_object(shape='plane') auto-sets is_static=True."""

    def test_plane_default_is_static(self, sim):
        r = sim._dispatch_action("add_object", {"name": "floor1", "shape": "plane"})
        assert r["status"] == "success"
        assert sim._world.objects["floor1"].is_static is True

    def test_plane_with_explicit_dynamic_errors(self, sim):
        r = sim._dispatch_action("add_object", {"name": "bad_floor", "shape": "plane", "is_static": False})
        assert r["status"] == "error"
        assert "plane" in r["content"][0]["text"].lower() and "is_static" in r["content"][0]["text"]


class TestSetGeomPropertiesAlias:
    """T28: set_geom_properties accepts the object name as a stand-in for the
    MJCF-injected '{name}_geom' geom name."""

    def test_object_name_resolves_to_geom(self, sim):
        sim._dispatch_action(
            "add_object",
            {"name": "box_alpha", "shape": "box", "size": [0.05, 0.05, 0.05]},
        )
        # Using the object name, not '{name}_geom', should work - the
        # T28 alias resolves to '{name}_geom' internally.
        r = sim._dispatch_action("set_geom_properties", {"geom_name": "box_alpha", "color": [1, 0, 0, 1]})
        # Success proves the alias resolved; error with 'Geom not found' would
        # mean T28 didn't kick in.
        assert r["status"] == "success", r
        assert "box_alpha" in r["content"][0]["text"] or "geom" in r["content"][0]["text"].lower()


class TestEvalPolicyDefaults:
    """T34: eval_policy resolves robot_name like run_policy; n_episodes default
    is 1."""

    def test_eval_policy_empty_world_errors(self, sim):
        # The fixture sim has a world but no robots; eval_policy reports the
        # empty scene rather than a "requires robot_name" guard.
        r = sim._dispatch_action("eval_policy", {})
        assert r["status"] == "error"
        assert "No robots" in r["content"][0]["text"]

    def test_eval_policy_unknown_robot_errors(self, sim):
        r = sim._dispatch_action("eval_policy", {"robot_name": "ghost"})
        assert r["status"] == "error"
        # Either "Robot X not found" (world has robots) or "No robots in sim"
        # (empty scene) - both are correct paths.
        text = r["content"][0]["text"]
        assert "ghost" in text or "No robots" in text


class TestRecordingStatusLifecycle:
    """T31: get_recording_status succeeds in every state (no world / not
    recording / recording) with distinguishing text."""

    def test_no_world_returns_success(self):
        s = Simulation(tool_name="rec_lifecycle_nw", mesh=False)
        try:
            r = s._dispatch_action("get_recording_status", {})
            assert r["status"] == "success"
            assert "No world" in r["content"][0]["text"]
        finally:
            s.cleanup()

    def test_not_recording_returns_success(self, sim):
        r = sim._dispatch_action("get_recording_status", {})
        assert r["status"] == "success"
        assert "Not recording" in r["content"][0]["text"]


class TestListRobotsPolicyStatus:
    """T37: list_robots reports per-robot policy status. Regression ward."""

    def test_list_robots_shows_idle_when_no_policy(self, sim):
        r = sim._dispatch_action("list_robots", {})
        assert r["status"] == "success"
        # No robots added, so we just expect the "No robots" message.
        assert "No robots" in r["content"][0]["text"]


class TestPolicyHorizonUnification:
    """T25: run_policy and start_policy accept n_steps (primary) / max_steps
    (legacy) as alternatives to duration. duration = n_steps / control_freq."""

    def test_run_policy_n_steps_zero_errors(self, sim):
        r = sim._dispatch_action("run_policy", {"robot_name": "ghost", "n_steps": 0})
        assert r["status"] == "error"
        # Either n_steps validation fires first, or robot-not-found; both are
        # acceptable error paths - we just want NO silent success.
        text = r["content"][0]["text"]
        assert ("n_steps" in text and "> 0" in text) or "Robot" in text

    def test_run_policy_negative_n_steps_errors(self, sim):
        r = sim._dispatch_action("run_policy", {"robot_name": "ghost", "n_steps": -10})
        assert r["status"] == "error"


class TestAddRobotDeprecation:
    """T22: the `name`-as-registry-fallback path emits a DeprecationWarning."""

    def test_add_robot_name_fallback_warns(self, sim):
        import warnings

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            # 'mock_never_registered' won't resolve to anything, so the
            # fallback is attempted but also fails.  We only care the
            # warning was triggered in the path.
            r = sim._dispatch_action("add_robot", {"name": "mock_never_registered"})
        # Either succeeded (name happened to resolve -> warning) or failed.
        # Just verify: if it succeeded via name fallback, a warning fired.
        warn_texts = [str(w.message) for w in captured if issubclass(w.category, DeprecationWarning)]
        if r["status"] == "success":
            assert any("deprecated" in t.lower() for t in warn_texts)


class TestMixedDataConfigRobots:
    """Regression: robots with different ``data_config`` values can coexist
    in one scene even when their MJCFs declare colliding nested default
    class names (e.g. ``<default class="visual">`` in both).

    Pre-fix, adding an ``h1`` humanoid after two ``so100`` arms errored with
    MuJoCo's *"repeated default class name"*. Fixed by per-config namespacing
    in scene_ops.
    """

    def test_two_arms_plus_humanoid_coexist(self, sim):
        r1 = sim.add_robot(name="alice", data_config="so100", position=[-0.6, 0, 0])
        assert r1["status"] == "success", r1["content"][0].get("text")
        r2 = sim.add_robot(name="bob", data_config="so100", position=[0.6, 0, 0])
        assert r2["status"] == "success", r2["content"][0].get("text")
        r3 = sim.add_robot(name="carol", data_config="h1", position=[0, 1.0, 0])
        assert r3["status"] == "success", r3["content"][0].get("text")
        assert set(sim._world.robots.keys()) == {"alice", "bob", "carol"}

    def test_four_different_configs_coexist(self, sim):
        specs = [
            ("alice", "so100", [-0.6, 0, 0]),
            ("bob", "so100", [0.6, 0, 0]),
            ("carol", "h1", [0, 1.0, 0]),
            ("dan", "panda", [0, -1.0, 0]),
        ]
        for name, cfg, pos in specs:
            r = sim.add_robot(name=name, data_config=cfg, position=pos)
            assert r["status"] == "success", f"add_robot({name}, {cfg}) failed: {r['content'][0].get('text')}"
        r = sim.step(n_steps=5)
        assert r["status"] == "success"
        # Ensure the physics actually advanced (forward kinematics would be
        # blocked by any lingering compile error).
        assert abs(sim._world.sim_time - 0.010) < 1e-9


class TestRemoveRobotActuallyRemoves:
    """Regression: remove_robot used to only pop the Python dict entry;
    the robot's MJCF bodies/actuators/sensors stayed in the compiled model.
    That blocked re-adding the same name and left stale DOFs consuming
    physics time per step.
    """

    def test_remove_robot_empties_model(self, sim):
        r = sim.add_robot(name="alice", data_config="so100")
        assert r["status"] == "success"
        njnt_before = sim._world._model.njnt
        assert njnt_before > 0, "precondition: robot should have added joints"

        r = sim.remove_robot(name="alice")
        assert r["status"] == "success"
        assert sim._world._model.njnt == 0
        assert sim._world._model.nbody == 1  # just the world root body
        assert "alice" not in sim._world.robots

    def test_readd_same_name_after_remove(self, sim):
        """Adding a robot, removing it, then adding again with the same name
        must succeed (MuJoCo rejects duplicate body names otherwise)."""
        assert sim.add_robot(name="alice", data_config="so100")["status"] == "success"
        assert sim.remove_robot(name="alice")["status"] == "success"
        r = sim.add_robot(name="alice", data_config="so100")
        assert r["status"] == "success", r["content"][0].get("text")
        assert sim._world._model.njnt == 6  # so100 has 6 joints

    def test_remove_middle_of_three_robots(self, sim):
        sim.add_robot(name="alice", data_config="so100", position=[-0.5, 0, 0])
        sim.add_robot(name="bob", data_config="so100", position=[0.5, 0, 0])
        sim.add_robot(name="carol", data_config="h1", position=[0, 1, 0])
        njnt_before = sim._world._model.njnt

        r = sim.remove_robot(name="bob")
        assert r["status"] == "success"
        assert set(sim._world.robots) == {"alice", "carol"}
        # bob was 6 joints; alice (6) + carol (19) = 25 should remain.
        assert sim._world._model.njnt == njnt_before - 6


class TestRemoveRobotPreservesState:
    """Regression tests for PR #85 follow-up (AGENTS.md "Per-name state
    copy, not flat index"): removing one robot must NOT reset the state
    of surviving robots or objects.

    Before the fix, ``eject_robot_from_scene`` rebuilt the scene from
    scratch and reset every remaining qpos/qvel to 0 (robots) or body
    pose (objects). Agents that called ``remove_robot`` mid-simulation
    silently lost physics state.
    """

    def test_surviving_robot_joint_state_is_preserved(self, sim):
        """After ``remove_robot(bob)``, alice's joints keep their qpos."""
        sim.add_robot(name="alice", data_config="so100", position=[-0.5, 0, 0])
        sim.add_robot(name="bob", data_config="so100", position=[0.5, 0, 0])

        # Drive alice's joints to a non-zero pose and step forward so
        # qpos actually reflects applied ctrl (not just a zero default).
        import numpy as np

        alice = sim._world.robots["alice"]
        # Write directly to qpos for a deterministic snapshot (avoids
        # ctrl-dynamics dependency). Each joint gets a distinct non-zero
        # value so accidental index shifts would be obvious.
        target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for jid, val in zip(alice.joint_ids, target):
            qadr = sim._world._model.jnt_qposadr[jid]
            sim._world._data.qpos[qadr] = val

        snapshot_before = np.array(
            [sim._world._data.qpos[sim._world._model.jnt_qposadr[jid]] for jid in alice.joint_ids]
        )

        # Remove bob - alice should survive with her joint state intact.
        r = sim.remove_robot(name="bob")
        assert r["status"] == "success"

        alice_after = sim._world.robots["alice"]
        snapshot_after = np.array(
            [sim._world._data.qpos[sim._world._model.jnt_qposadr[jid]] for jid in alice_after.joint_ids]
        )

        np.testing.assert_allclose(
            snapshot_before,
            snapshot_after,
            atol=1e-10,
            err_msg="alice's joint qpos was reset by remove_robot(bob)",
        )

    def test_surviving_object_freejoint_pose_is_preserved(self, sim):
        """An object's freejoint qpos (position + quat) survives the
        eject rebuild. Before the fix, objects snapped back to their
        ``add_object`` spawn pose."""
        import numpy as np

        sim.add_robot(name="alice", data_config="so100")
        sim.add_robot(name="bob", data_config="so100", position=[1, 0, 0])
        sim.add_object(name="cube", shape="box", size=[0.05, 0.05, 0.05], position=[0.3, 0, 0.05])

        mj = sim._mj
        cube_jid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_JOINT, "cube_joint")
        assert cube_jid >= 0, "cube freejoint must exist"
        cube_qadr = int(sim._world._model.jnt_qposadr[cube_jid])

        # Move the cube to a distinct position + tilted orientation so
        # the test fails loudly if state is reset to spawn pose.
        new_pose = [0.7, -0.2, 0.15, 0.7071, 0.0, 0.7071, 0.0]
        for i, v in enumerate(new_pose):
            sim._world._data.qpos[cube_qadr + i] = v
        mj.mj_forward(sim._world._model, sim._world._data)

        # Eject bob.
        r = sim.remove_robot(name="bob")
        assert r["status"] == "success"

        # Cube freejoint must still be there at the moved pose.
        cube_jid2 = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_JOINT, "cube_joint")
        assert cube_jid2 >= 0, "cube freejoint disappeared after remove_robot"
        cube_qadr2 = int(sim._world._model.jnt_qposadr[cube_jid2])
        restored = np.array([sim._world._data.qpos[cube_qadr2 + i] for i in range(7)])
        np.testing.assert_allclose(
            restored,
            new_pose,
            atol=1e-10,
            err_msg="cube freejoint pose was reset by remove_robot",
        )

    def test_ejected_robot_state_is_not_restored(self, sim):
        """The ejected robot's joints should NOT appear in the new model.
        This is the contrapositive of the preservation tests - confirms
        the snapshot/restore loop only touches surviving joints.
        """
        sim.add_robot(name="alice", data_config="so100")
        sim.add_robot(name="bob", data_config="so100", position=[1, 0, 0])

        bob = sim._world.robots["bob"]
        bob_prefix = bob.namespace  # e.g. "bob/"
        assert bob_prefix, "bob must have a namespace for this test to be meaningful"

        r = sim.remove_robot(name="bob")
        assert r["status"] == "success"

        mj = sim._mj
        # No joint under bob's prefix should exist anymore.
        model = sim._world._model
        for jid in range(model.njnt):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
            assert name is None or not name.startswith(bob_prefix), f"ejected robot's joint survived: {name!r}"
