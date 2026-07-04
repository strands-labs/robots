"""Input validation regression tests for PR #85 fixes (T7, T9, T10).

These guard against silent data-integrity bugs and process-killing MuJoCo
aborts that were caught by autonomous local testing on PR #85.
"""

import pytest

pytest.importorskip("mujoco")
from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No GL context available (headless CI without EGL/OSMesa)",
)

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim_with_world():
    """A minimal simulation with an empty world for validation tests."""
    sim = Simulation()
    sim.create_world()
    yield sim
    sim.destroy()


@pytest.fixture
def sim_with_robot():
    """A simulation with a single robot for physics-validation tests."""
    sim = Simulation()
    sim.create_world()
    # Use a built-in registry robot - no network I/O
    res = sim.add_robot(name="panda", data_config="panda")
    if res["status"] != "success":
        pytest.skip(f"panda not available: {res['content'][0]['text']}")
    sim.reset()
    yield sim
    sim.destroy()


# T9: step validation


class TestStepValidation:
    def test_step_negative_errors(self, sim_with_world):
        """step(n_steps=-5) must error and NOT decrement step_count."""
        initial = sim_with_world._world.step_count
        res = sim_with_world.step(n_steps=-5)
        assert res["status"] == "error"
        assert "n_steps must be >= 0" in res["content"][0]["text"]
        assert sim_with_world._world.step_count == initial, "step_count must not change on rejected call"

    def test_step_zero_is_noop(self, sim_with_world):
        """step(n_steps=0) is a successful no-op."""
        initial = sim_with_world._world.step_count
        res = sim_with_world.step(n_steps=0)
        assert res["status"] == "success"
        assert "no-op" in res["content"][0]["text"].lower()
        assert sim_with_world._world.step_count == initial

    def test_step_positive_still_works(self, sim_with_world):
        """Baseline: non-negative n_steps continues to work."""
        res = sim_with_world.step(n_steps=3)
        assert res["status"] == "success"
        assert sim_with_world._world.step_count == 3

    def test_step_float_is_coerced_to_int(self, sim_with_world):
        """A non-int but coercible n_steps (e.g. 3.0) is accepted as int(3)."""
        res = sim_with_world.step(n_steps=3.0)
        assert res["status"] == "success"
        assert sim_with_world._world.step_count == 3

    def test_step_non_coercible_type_errors(self, sim_with_world):
        """A non-int n_steps that cannot be coerced errors and names the type."""
        initial = sim_with_world._world.step_count
        res = sim_with_world.step(n_steps="not-a-number")
        assert res["status"] == "error"
        msg = res["content"][0]["text"]
        assert "n_steps must be an integer" in msg
        assert "str" in msg
        assert sim_with_world._world.step_count == initial, "step_count must not change on rejected call"

    def test_step_exceeds_max_per_call_errors(self, sim_with_world):
        """n_steps above the per-call ceiling is rejected without stepping."""
        initial = sim_with_world._world.step_count
        over = sim_with_world._MAX_STEPS_PER_CALL + 1
        res = sim_with_world.step(n_steps=over)
        assert res["status"] == "error"
        assert "exceeds max" in res["content"][0]["text"]
        assert sim_with_world._world.step_count == initial, "step_count must not change on rejected call"


# Raycast zero-direction guard


class TestRaycastValidation:
    def test_zero_direction_errors_not_crash(self, sim_with_robot):
        """raycast with zero direction used to abort the interpreter. Now errors cleanly."""
        res = sim_with_robot.raycast(origin=[0, 0, 1], direction=[0, 0, 0])
        assert res["status"] == "error"
        assert "zero-length" in res["content"][0]["text"].lower()

    def test_wrong_length_direction_errors(self, sim_with_robot):
        res = sim_with_robot.raycast(origin=[0, 0, 1], direction=[0, 0])
        assert res["status"] == "error"
        assert "3 elements" in res["content"][0]["text"]

    def test_wrong_length_origin_errors(self, sim_with_robot):
        res = sim_with_robot.raycast(origin=[0, 0], direction=[0, 0, 1])
        assert res["status"] == "error"
        assert "3 elements" in res["content"][0]["text"]

    def test_valid_raycast_still_works(self, sim_with_robot):
        res = sim_with_robot.raycast(origin=[0, 0, 5], direction=[0, 0, -1])
        assert res["status"] == "success"

    def test_multi_raycast_zero_direction_isolates_error(self, sim_with_robot):
        """A zero-length direction in one ray must not abort the whole batch."""
        res = sim_with_robot.multi_raycast(
            origin=[0, 0, 5],
            directions=[[0, 0, -1], [0, 0, 0], [1, 0, -1]],
        )
        assert res["status"] == "success"
        # The JSON payload should show error on ray[1] only
        rays = res["content"][1]["json"]["rays"]
        assert len(rays) == 3
        assert rays[1].get("error") is not None
        assert "zero-length" in rays[1]["error"]


class TestApplyForceValidation:
    def test_missing_both_force_and_torque_errors(self, sim_with_robot):
        """apply_force(body='link1') with no force/torque must error, not silent success."""
        res = sim_with_robot.apply_force(body_name="link1")
        assert res["status"] == "error"
        assert "at least one" in res["content"][0]["text"].lower()

    def test_explicit_zero_force_still_clears_latched(self, sim_with_robot):
        """Regression: apply_force(body, force=[0,0,0]) is the documented way to clear."""
        # First latch a force
        r1 = sim_with_robot.apply_force(body_name="link1", force=[10, 0, 0])
        assert r1["status"] == "success"
        # Then clear with explicit zero - this MUST remain valid
        r2 = sim_with_robot.apply_force(body_name="link1", force=[0, 0, 0])
        assert r2["status"] == "success"

    def test_wrong_length_force_errors(self, sim_with_robot):
        res = sim_with_robot.apply_force(body_name="link1", force=[1, 2])
        assert res["status"] == "error"
        assert "3-element" in res["content"][0]["text"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            # non-numeric elements: pre-fix these raise ValueError inside
            # np.array(dtype=float64), escaping the structured-error contract.
            {"force": ["a", "b", "c"]},
            {"torque": ["x", "y", "z"]},
            {"force": [0.0, 0.0, 0.0], "point": ["p", "q", "r"]},
            # nested lists are length-3 but not scalar numbers.
            {"force": [[1], [2], [3]]},
            # None element is neither a number nor iterable of length mismatch.
            {"force": [1, None, 3]},
        ],
    )
    def test_non_numeric_elements_error(self, sim_with_robot, kwargs):
        """Regression: 3-element vectors with non-numeric entries must return a
        structured error, not raise or silently coerce into the physics buffer."""
        res = sim_with_robot.apply_force(body_name="link1", **kwargs)
        assert res["status"] == "error"
        assert "must be numbers" in res["content"][0]["text"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            # nan/inf pass the length check and, pre-fix, are silently applied to
            # qfrc_applied - poisoning every subsequent mj_step.
            {"force": [float("nan"), 0.0, 0.0]},
            {"force": [float("inf"), 0.0, 0.0]},
            {"torque": [0.0, float("-inf"), 0.0]},
            {"force": [0.0, 0.0, 0.0], "point": [0.0, 0.0, float("nan")]},
        ],
    )
    def test_non_finite_elements_error(self, sim_with_robot, kwargs):
        """Regression: nan/inf in a force/torque/point vector must return a
        structured error instead of injecting non-finite state into physics."""
        res = sim_with_robot.apply_force(body_name="link1", **kwargs)
        assert res["status"] == "error"
        assert "finite numbers" in res["content"][0]["text"]


# negative/invalid mass, timestep


class TestMassAndTimestepValidation:
    def test_set_body_properties_negative_mass_errors(self, sim_with_robot):
        res = sim_with_robot.set_body_properties(body_name="link1", mass=-1.0)
        assert res["status"] == "error"
        assert "must be > 0" in res["content"][0]["text"]

    def test_set_body_properties_zero_mass_errors(self, sim_with_robot):
        res = sim_with_robot.set_body_properties(body_name="link1", mass=0.0)
        assert res["status"] == "error"

    def test_set_body_properties_positive_mass_works(self, sim_with_robot):
        res = sim_with_robot.set_body_properties(body_name="link1", mass=2.5)
        assert res["status"] == "success"

    def test_set_timestep_negative_errors(self, sim_with_world):
        res = sim_with_world.set_timestep(-0.01)
        assert res["status"] == "error"
        assert "positive" in res["content"][0]["text"]

    def test_set_timestep_zero_errors(self, sim_with_world):
        res = sim_with_world.set_timestep(0)
        assert res["status"] == "error"

    def test_set_timestep_positive_works(self, sim_with_world):
        res = sim_with_world.set_timestep(0.001)
        assert res["status"] == "success"

    def test_set_timestep_large_warns_but_succeeds(self, sim_with_world):
        res = sim_with_world.set_timestep(0.5)
        assert res["status"] == "success"
        assert "Warning:" in res["content"][0]["text"] and "unusually" in res["content"][0]["text"]

    def test_set_timestep_nan_errors(self, sim_with_world):
        """NaN must not pass the positivity guard (nan <= 0 is False)."""

        res = sim_with_world.set_timestep(float("nan"))
        assert res["status"] == "error"
        assert "finite" in res["content"][0]["text"]

    def test_set_timestep_inf_errors(self, sim_with_world):
        res = sim_with_world.set_timestep(float("inf"))
        assert res["status"] == "error"
        assert "finite" in res["content"][0]["text"]

    def test_set_gravity_nan_errors(self, sim_with_world):
        """NaN in gravity components must be rejected."""
        res = sim_with_world.set_gravity([0, 0, float("nan")])
        assert res["status"] == "error"
        assert "finite" in res["content"][0]["text"]

    def test_set_gravity_inf_errors(self, sim_with_world):
        res = sim_with_world.set_gravity([0, float("inf"), -9.81])
        assert res["status"] == "error"
        assert "finite" in res["content"][0]["text"]

    def test_set_gravity_scalar_nan_errors(self, sim_with_world):
        """Scalar NaN must be rejected even though it goes through the
        `[0, 0, scalar]` expansion path before the finite check.

        Regression guard: prevents someone from accidentally moving the
        isfinite loop ahead of the scalar expansion and re-opening the gap.
        """
        res = sim_with_world.set_gravity(float("nan"))
        assert res["status"] == "error"
        assert "finite" in res["content"][0]["text"]

    def test_set_gravity_scalar_inf_errors(self, sim_with_world):
        """Scalar ±Inf must also be rejected through the expansion path."""
        res = sim_with_world.set_gravity(float("inf"))
        assert res["status"] == "error"
        assert "finite" in res["content"][0]["text"]


# set_gravity dim validation


class TestSetGravityValidation:
    def test_two_element_gravity_errors(self, sim_with_world):
        res = sim_with_world.set_gravity([0.0, 0.0])
        assert res["status"] == "error"
        assert "3-element" in res["content"][0]["text"]

    def test_scalar_gravity_still_works(self, sim_with_world):
        # Scalar form convenience (z-only) preserved
        res = sim_with_world.set_gravity(-9.81)
        assert res["status"] == "success"

    def test_full_vector_gravity_works(self, sim_with_world):
        res = sim_with_world.set_gravity([1.0, 2.0, -9.0])
        assert res["status"] == "success"


# set_joint_positions list/dict support


class TestSetJointPositionsForms:
    def test_dict_form_works(self, sim_with_robot):
        # Pick a valid joint name from the robot
        joint_names = list(sim_with_robot._world.robots.values())[0].joint_names or []
        if not joint_names:
            import pytest as _pytest

            _pytest.skip("robot has no named joints")
        res = sim_with_robot.set_joint_positions(positions={joint_names[0]: 0.1})
        assert res["status"] == "success"

    def test_list_form_matches_count(self, sim_with_robot):
        joint_names = list(sim_with_robot._world.robots.values())[0].joint_names or []
        if not joint_names:
            import pytest as _pytest

            _pytest.skip("robot has no named joints")
        res = sim_with_robot.set_joint_positions(positions=[0.0] * len(joint_names))
        assert res["status"] == "success", res["content"][0]["text"]

    def test_list_form_wrong_length_errors(self, sim_with_robot):
        # 999 is almost certainly wrong for any robot
        res = sim_with_robot.set_joint_positions(positions=[0.1] * 999)
        assert res["status"] == "error"
        assert "does not match" in res["content"][0]["text"]


# set_joint_velocities list/dict support


class TestSetJointVelocitiesForms:
    """List/dict dispatch parity with set_joint_positions.

    set_joint_velocities accepts the same dict and positional-list forms as
    set_joint_positions (it documents "see set_joint_positions for list
    semantics"). These pin the list-form dispatch branches: count match, count
    mismatch, multi-robot ambiguity, an empty world, robot_name selection, the
    non-dict/list type guard, and unknown-joint skipping.
    """

    def test_dict_form_sets_qvel(self, sim_with_robot):
        joint_names = list(sim_with_robot._world.robots.values())[0].joint_names or []
        if not joint_names:
            pytest.skip("robot has no named joints")
        res = sim_with_robot.set_joint_velocities(velocities={joint_names[0]: 0.5})
        assert res["status"] == "success"
        assert "1/1" in res["content"][0]["text"]

    def test_list_form_matches_count(self, sim_with_robot):
        joint_names = list(sim_with_robot._world.robots.values())[0].joint_names or []
        if not joint_names:
            pytest.skip("robot has no named joints")
        res = sim_with_robot.set_joint_velocities(velocities=[0.1] * len(joint_names))
        assert res["status"] == "success", res["content"][0]["text"]
        assert f"{len(joint_names)}/{len(joint_names)}" in res["content"][0]["text"]

    def test_list_form_wrong_length_errors(self, sim_with_robot):
        res = sim_with_robot.set_joint_velocities(velocities=[0.1] * 999)
        assert res["status"] == "error"
        assert "does not match" in res["content"][0]["text"]

    def test_list_form_with_no_robot_errors(self, sim_with_world):
        """List form needs a robot to map positions onto joint names."""
        res = sim_with_world.set_joint_velocities(velocities=[0.1])
        assert res["status"] == "error"
        assert "requires a robot" in res["content"][0]["text"]

    def test_list_form_multi_robot_is_ambiguous(self, sim_with_world):
        """With >1 robot and no robot_name, the list form refuses to guess."""
        if sim_with_world.add_robot(name="panda", data_config="panda")["status"] != "success":
            pytest.skip("panda not available")
        if sim_with_world.add_robot(name="so100", data_config="so100", position=[1, 0, 0])["status"] != "success":
            pytest.skip("so100 not available")
        sim_with_world.reset()
        res = sim_with_world.set_joint_velocities(velocities=[0.1] * 9)
        assert res["status"] == "error"
        assert "ambiguous" in res["content"][0]["text"]

    def test_list_form_robot_name_selects_target(self, sim_with_world):
        """robot_name disambiguates which robot the list maps onto."""
        if sim_with_world.add_robot(name="panda", data_config="panda")["status"] != "success":
            pytest.skip("panda not available")
        if sim_with_world.add_robot(name="so100", data_config="so100", position=[1, 0, 0])["status"] != "success":
            pytest.skip("so100 not available")
        sim_with_world.reset()
        panda_joints = sim_with_world._world.robots["panda"].joint_names
        res = sim_with_world.set_joint_velocities(velocities=[0.1] * len(panda_joints), robot_name="panda")
        assert res["status"] == "success", res["content"][0]["text"]

    def test_list_form_unknown_robot_name_errors(self, sim_with_robot):
        res = sim_with_robot.set_joint_velocities(velocities=[0.1], robot_name="ghost")
        assert res["status"] == "error"
        assert "not found" in res["content"][0]["text"]

    def test_non_dict_or_list_type_rejected(self, sim_with_robot):
        res = sim_with_robot.set_joint_velocities(velocities=5)
        assert res["status"] == "error"
        assert "must be a dict or list" in res["content"][0]["text"]

    def test_dict_form_unknown_joint_is_skipped_not_fatal(self, sim_with_robot):
        """Unknown joint names are reported as ignored, not raised."""
        res = sim_with_robot.set_joint_velocities(velocities={"nope": 1.0})
        assert res["status"] == "success"
        assert "0/1" in res["content"][0]["text"]
        assert "nope" in res["content"][0]["text"]


# Policy-running guards


class TestPolicyRunningGuards:
    """Simulate policy-running state by poisoning _policy_threads.

    We insert a fake Future whose done() returns False so _require_no_running_policy
    flags a running policy without actually starting one.
    """

    def _install_fake_running_policy(self, sim):
        class _FakeRunningFuture:
            def done(self):
                return False

        sim._policy_threads["fake"] = _FakeRunningFuture()

    def test_reset_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.reset()
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_set_gravity_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.set_gravity([0, 0, -5])
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_set_timestep_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.set_timestep(0.001)
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_set_joint_positions_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.set_joint_positions(positions={"nope": 0.0})
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_apply_force_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.apply_force(body_name="link1", force=[1, 0, 0])
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_set_body_properties_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.set_body_properties(body_name="link1", mass=3.0)
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]

    def test_randomize_blocked(self, sim_with_robot):
        self._install_fake_running_policy(sim_with_robot)
        res = sim_with_robot.randomize(seed=42)
        assert res["status"] == "error"
        assert "while a policy is running" in res["content"][0]["text"]


# add_robot initial state is zero


class TestAddRobotInitialState:
    """After add_robot, qpos/qvel/ctrl must be zero without needing reset()."""

    def test_initial_qpos_is_zero(self):
        import numpy as np

        sim = Simulation()
        try:
            sim.create_world()
            res = sim.add_robot(name="panda", data_config="panda")
            if res["status"] != "success":
                import pytest as _pytest

                _pytest.skip(f"panda not available: {res['content'][0]['text']}")
            # IMPORTANT: do NOT call reset. T6 requires that add_robot itself leaves a clean state.
            data = sim._world._data
            assert np.allclose(data.qpos, 0.0), f"qpos should be zero after add_robot, got {data.qpos}"
            assert np.allclose(data.qvel, 0.0), f"qvel should be zero after add_robot, got {data.qvel}"
            assert np.allclose(data.ctrl, 0.0), f"ctrl should be zero after add_robot, got {data.ctrl}"
        finally:
            sim.destroy()


# render camera strict validation


@requires_gl
class TestRenderCameraValidation:
    def test_unknown_camera_errors(self, sim_with_world):
        res = sim_with_world.render(camera_name="does_not_exist", width=64, height=48)
        assert res["status"] == "error"
        assert "not found" in res["content"][0]["text"]

    def test_default_camera_labelled_honestly(self, sim_with_world):
        res = sim_with_world.render(camera_name="default", width=64, height=48)
        if res["status"] != "success":
            import pytest as _pytest

            _pytest.skip(f"offscreen render unavailable: {res['content'][0]['text']}")
        assert "free (default)" in res["content"][0]["text"]

    def test_free_alias_labelled_honestly(self, sim_with_world):
        res = sim_with_world.render(camera_name="free", width=64, height=48)
        if res["status"] != "success":
            import pytest as _pytest

            _pytest.skip(f"offscreen render unavailable: {res['content'][0]['text']}")
        assert "free (default)" in res["content"][0]["text"]

    def test_render_depth_unknown_camera_errors(self, sim_with_world):
        res = sim_with_world.render_depth(camera_name="ghost_cam", width=64, height=48)
        assert res["status"] == "error"
        assert "not found" in res["content"][0]["text"]


# camera target actually applied


class TestAddCameraTargetOrients:
    """The 'headline broken feature': add_camera(target=...) was silently dropped
    so every custom camera rendered the same default view. These tests verify
    that orientation now flows through to the rendered pixels.
    """

    def _with_obj(self):
        """Create a world with a distinguishable colored object for the cameras to frame."""
        sim = Simulation()
        sim.create_world()
        # Add a vivid red box at origin to make camera differences visible.
        sim.add_object(
            name="target_box",
            shape="box",
            size=[0.3, 0.3, 0.3],
            position=[0.0, 0.0, 0.25],
            color=[1.0, 0.0, 0.0, 1.0],
            is_static=True,
        )
        return sim

    def test_degenerate_target_equals_position_errors(self):
        sim = self._with_obj()
        try:
            res = sim.add_camera(name="bad_cam", position=[1, 2, 3], target=[1, 2, 3])
            assert res["status"] == "error"
            assert "identical" in res["content"][0]["text"]
        finally:
            sim.destroy()

    def test_wrong_length_position_errors(self):
        sim = self._with_obj()
        try:
            res = sim.add_camera(name="bad_cam", position=[1, 2], target=[0, 0, 0])
            assert res["status"] == "error"
            assert "3 elements" in res["content"][0]["text"]
        finally:
            sim.destroy()

    def test_camera_orientation_written(self):
        """A target'd camera must end up with a non-default orientation in the
        compiled model. Previously this test asserted on the raw ``xyaxes="..."``
        attribute in the scene XML, which the MjSpec builder path replaces with
        a ``quat`` attribute. Both representations resolve to the same rotation
        matrix in the compiled MjModel (``cam_mat0``) - which is what we
        actually care about for rendering.
        """
        import numpy as np

        sim = self._with_obj()
        try:
            res = sim.add_camera(name="side_cam", position=[2.0, 0.0, 0.3], target=[0.0, 0.0, 0.25])
            assert res["status"] == "success", res["content"][0]["text"]

            mj = sim._mj
            model = sim._world._model
            assert model is not None
            cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "side_cam")
            assert cam_id >= 0, "camera was not registered in compiled model"

            # MuJoCo's default camera orientation is identity (looks along -Z).
            # Our target->quat conversion for position [2, 0, 0.3] looking at
            # [0, 0, 0.25] must produce a non-identity rotation.
            rot = model.cam_mat0[cam_id].reshape(3, 3)
            assert not np.allclose(rot, np.eye(3)), "camera has default (identity) orientation - target was ignored"
        finally:
            sim.destroy()

    def test_different_targets_produce_different_orientations(self):
        """Two cameras at the SAME position but different targets must produce
        DIFFERENT rotation matrices in the compiled MjModel. Before the
        camera-target fix (T* in PR #85) both cameras shared MuJoCo's default
        look direction, so rendered frames were identical regardless of the
        ``target`` argument.

        We assert on ``cam_mat0`` (the rotation matrix of the camera frame
        at qpos0) rather than rendered pixels, because offscreen GL on some
        CI runners produces blank frames and makes pixel comparison
        unreliable. cam_mat0 is representation-agnostic - works under both
        legacy MJCFBuilder (xyaxes-based) and SpecBuilder (quat-based) paths.
        """
        import numpy as np

        sim = self._with_obj()
        try:
            res_a = sim.add_camera(name="cam_a", position=[2.0, 0.0, 0.5], target=[0.0, 0.0, 0.25])
            res_b = sim.add_camera(name="cam_b", position=[2.0, 0.0, 0.5], target=[0.0, 2.0, 0.25])
            assert res_a["status"] == "success"
            assert res_b["status"] == "success"

            mj = sim._mj
            model = sim._world._model
            a_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "cam_a")
            b_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "cam_b")
            assert a_id >= 0 and b_id >= 0

            rot_a = model.cam_mat0[a_id].reshape(3, 3)
            rot_b = model.cam_mat0[b_id].reshape(3, 3)
            assert not np.allclose(rot_a, rot_b, atol=1e-3), (
                "cameras with different targets must have different orientations "
                "(their cam_mat0 rotation matrices are currently identical, which means "
                "`target` is being ignored)."
            )
        finally:
            sim.destroy()


class TestAddCameraParamValidation:
    """add_camera validates fov / width / height eagerly.

    Previously add_camera validated position, target, duplicate names and the
    parent_body mount, but accepted any fov and any render resolution. A
    non-positive or out-of-range fov silently registered a degenerate camera
    (fov <= 0) or aborted with a cryptic "spec recompile refused" (fov >= 180),
    and a non-positive / oversized width or height was accepted at config time
    only to fail with an opaque GL/Renderer error on the first render. These
    guard the config-time rejection with an actionable message.
    """

    def test_fov_zero_errors(self, sim_with_world):
        res = sim_with_world.add_camera(name="c", position=[0.5, 0, 0.3], target=[0.2, 0, 0.05], fov=0.0)
        assert res["status"] == "error"
        assert "fov" in res["content"][0]["text"]
        assert "c" not in sim_with_world._world.cameras

    def test_fov_negative_errors(self, sim_with_world):
        res = sim_with_world.add_camera(name="c", position=[0.5, 0, 0.3], target=[0.2, 0, 0.05], fov=-10.0)
        assert res["status"] == "error"
        assert "(0, 180)" in res["content"][0]["text"]

    def test_fov_at_180_errors(self, sim_with_world):
        res = sim_with_world.add_camera(name="c", position=[0.5, 0, 0.3], target=[0.2, 0, 0.05], fov=180.0)
        assert res["status"] == "error"
        assert "fov" in res["content"][0]["text"]

    def test_fov_above_180_errors(self, sim_with_world):
        res = sim_with_world.add_camera(name="c", position=[0.5, 0, 0.3], target=[0.2, 0, 0.05], fov=250.0)
        assert res["status"] == "error"
        # must fail cleanly, NOT with the opaque spec-recompile message
        assert "spec recompile refused" not in res["content"][0]["text"]

    def test_fov_nan_errors(self, sim_with_world):
        res = sim_with_world.add_camera(name="c", position=[0.5, 0, 0.3], target=[0.2, 0, 0.05], fov=float("nan"))
        assert res["status"] == "error"
        assert "finite" in res["content"][0]["text"]

    def test_fov_inf_errors(self, sim_with_world):
        res = sim_with_world.add_camera(name="c", position=[0.5, 0, 0.3], target=[0.2, 0, 0.05], fov=float("inf"))
        assert res["status"] == "error"
        assert "finite" in res["content"][0]["text"]

    def test_fov_bool_rejected(self, sim_with_world):
        # bool is an int subclass; True (== 1 deg) is almost certainly a caller bug.
        res = sim_with_world.add_camera(name="c", position=[0.5, 0, 0.3], target=[0.2, 0, 0.05], fov=True)
        assert res["status"] == "error"
        assert "finite number" in res["content"][0]["text"]

    def test_width_zero_errors(self, sim_with_world):
        res = sim_with_world.add_camera(name="c", position=[0.5, 0, 0.3], target=[0.2, 0, 0.05], width=0)
        assert res["status"] == "error"
        assert "> 0" in res["content"][0]["text"]
        assert "c" not in sim_with_world._world.cameras

    def test_height_negative_errors(self, sim_with_world):
        res = sim_with_world.add_camera(name="c", position=[0.5, 0, 0.3], target=[0.2, 0, 0.05], height=-5)
        assert res["status"] == "error"
        assert "> 0" in res["content"][0]["text"]

    def test_width_above_abs_max_errors(self, sim_with_world):
        res = sim_with_world.add_camera(name="c", position=[0.5, 0, 0.3], target=[0.2, 0, 0.05], width=100000)
        assert res["status"] == "error"
        assert "add_camera" in res["content"][0]["text"]
        assert "render:" not in res["content"][0]["text"]

    def test_valid_camera_still_added(self, sim_with_world):
        res = sim_with_world.add_camera(
            name="good", position=[0.5, 0, 0.3], target=[0.2, 0, 0.05], fov=58.0, width=256, height=256
        )
        assert res["status"] == "success"
        assert "good" in sim_with_world._world.cameras


# send_action ordered-vector normalization


class TestSendActionVectorValidation:
    """Pin send_action's ordered-vector contract.

    send_action accepts either a ``{name: value}`` mapping or an ordered numeric
    vector aligned with ``robot_action_keys``. A vector with a non-numeric entry
    (e.g. a stray string) must fail with a structured error naming the offending
    conversion, not raise past the tool boundary or coerce garbage downstream.
    """

    def test_non_numeric_entry_returns_structured_error(self, sim_with_robot):
        keys = sim_with_robot.robot_action_keys("panda")
        assert len(keys) >= 2, keys
        bad_vector = [0.0, "not_a_number", *([0.0] * (len(keys) - 2))]
        res = sim_with_robot.send_action(bad_vector)
        assert res["status"] == "error", res
        assert "non-numeric entry" in res["content"][0]["text"]

    def test_numeric_vector_still_applies(self, sim_with_robot):
        keys = sim_with_robot.robot_action_keys("panda")
        res = sim_with_robot.send_action([0.0] * len(keys))
        assert res["status"] == "success", res["content"][0]["text"]
