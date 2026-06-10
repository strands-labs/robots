"""End-to-end integration suite, one test per user journey.

Design principles

* **Journey-per-test**: each test executes a realistic user sequence end-to-end
  (scene build → physics probe → policy rollout → teardown). No mocks for the
  simulator itself - only the few optional dependencies (HF dataset) get
  lightweight fakes where shipping a real dataset would be wasteful.

* **One sim instance per test**: we *destroy* at the end so tests are
  independent, but within a journey we reuse the same ``Simulation`` object
  to exercise state transitions (reset, save/load, policy start/stop).

* **No coverage scaffolding**: every assertion is a user-visible invariant
  (status == "success", delta > threshold, file exists), not an internal
  implementation detail.

* **Fast path**: every test that doesn't need a real VLM uses the
  ``MockPolicy`` + ``so101`` registered robot. Total suite wall time target:
  < 30 s on MPS.

Coverage targets

These 10 journeys together touch every tool_spec action that's worth
exercising, every public method on ``Simulation`` + mixins, and every
``PolicyRunner`` entry point (``run``/``replay``/``evaluate``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "glfw")

# Shared fixtures


@pytest.fixture
def sim():
    """A fresh MuJoCo simulation with one so101 arm. Destroyed on teardown."""
    from strands_robots.simulation import Simulation

    s = Simulation()
    s.create_world(timestep=0.002, gravity=[0, 0, -9.81])
    s.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    s.add_camera("cam_front", position=[0.0, -0.5, 0.3], target=[0.0, 0.2, 0.1])
    s.step(n_steps=10)  # settle
    yield s
    s.destroy()


@pytest.fixture
def mock_policy(sim):
    """A ready-to-use MockPolicy with state keys bound to ``sim``'s robot."""
    from strands_robots.policies import MockPolicy

    p = MockPolicy()
    p.set_robot_state_keys(sim.robot_joint_names("arm"))
    return p


def _content_texts(result):
    """Pull every text block from a tool result - used in assertions."""
    return " ".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))


def _content_json(result, idx=1):
    """Schema-tolerant extraction of the structured JSON content block."""
    block = result["content"][idx]
    if "json" in block:
        return block["json"]
    return json.loads(block["text"])


def _n_images(result):
    return sum(1 for c in result.get("content", []) if isinstance(c, dict) and "image" in c)


# J1 · SCENE BUILD - multi-robot, multi-object, multi-camera composition


def test_j1_scene_build_multi_robot_multi_camera():
    """Build a 3-arm / 3-object / 4-camera scene → every sim invariant holds.

    Exercises: ``create_world``, ``add_robot`` ×3 (multi-robot asset merge),
    ``add_object`` ×3 (primitive shapes), ``add_camera`` ×4, ``step``,
    ``render_all`` (single-shot multi-view), ``list_robots``/``list_objects``,
    ``get_features`` (introspection + ``{"json":{...}}`` schema).
    """
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world(timestep=0.002)

    # 3 so101 arms spaced on X
    for i, x in enumerate([-0.4, 0.0, 0.4], start=1):
        r = sim.add_robot(f"arm_{i}", data_config="so101", position=[x, 0, 0])
        assert r["status"] == "success", r

    # 3 primitive objects covering each code path in _object_xml
    shapes = [
        ("red_cube", "box", [0.025, 0.025, 0.025], [-0.2, 0.25, 0.05], [1, 0, 0, 1]),
        ("blue_ball", "sphere", [0.03, 0.03, 0.03], [0.0, 0.25, 0.05], [0, 0, 1, 1]),
        ("green_rod", "cylinder", [0.02, 0.02, 0.08], [0.2, 0.25, 0.08], [0, 1, 0, 1]),
    ]
    for name, shape, size, pos, rgba in shapes:
        r = sim.add_object(name=name, shape=shape, size=size, position=pos, rgba=rgba)
        assert r["status"] == "success", r

    # 4 user-defined cameras
    cams = [
        ("overhead", [0, 0, 0.9], [0, 0.2, 0]),
        ("front", [0, -0.6, 0.3], [0, 0.2, 0.1]),
        ("left", [-0.5, 0, 0.4], [0, 0, 0.1]),
        ("right", [0.5, 0, 0.4], [0, 0, 0.1]),
    ]
    for name, p, t in cams:
        r = sim.add_camera(name=name, position=p, target=t)
        assert r["status"] == "success", r

    sim.step(n_steps=20)

    # Invariants
    assert sorted(sim.list_robots()) == ["arm_1", "arm_2", "arm_3"]

    lst = sim.list_objects()
    assert lst["status"] == "success"
    assert all(n in _content_texts(lst) for n in ["red_cube", "blue_ball", "green_rod"])

    # render_all should emit one image block per user camera (plus any
    # model-defined ones like wrist cams). We bound only the lower limit.
    views = sim.render_all(width=64, height=48)
    assert views["status"] == "success"
    assert _n_images(views) >= 4  # our 4 user cams, at minimum

    # get_features emits a structured-JSON block
    feats = sim.get_features()
    assert feats["status"] == "success"
    data = _content_json(feats, idx=1)["features"]
    assert data["n_joints"] >= 3 * 6  # so101 has 6 DoF
    assert data["n_actuators"] >= 3 * 6
    assert set(data["robots"]) == {"arm_1", "arm_2", "arm_3"}

    sim.destroy()


# J2 · PHYSICS PROBE - every physics introspection method on a live sim


def test_j2_physics_probe_every_mixin_method(sim):
    """Hit every ``PhysicsMixin`` method and ``RenderingMixin.get_contacts``.

    Exercises: ``apply_force``, ``raycast``, ``multi_raycast``,
    ``get_jacobian``, ``get_mass_matrix``, ``get_energy``, ``get_total_mass``,
    ``get_body_state``, ``get_sensor_data``, ``get_contacts``,
    ``get_contact_forces``, ``inverse_dynamics``, ``forward_kinematics``,
    ``set_body_properties``, ``set_geom_properties``, ``set_joint_velocities``.
    """
    sim.add_object("target", shape="box", size=[0.03, 0.03, 0.03], position=[0.2, 0.2, 0.05])
    sim.step(n_steps=10)

    # apply_force → target should gain KE
    e_before = _content_json(sim.get_energy())["kinetic"]
    r = sim.apply_force(body_name="target", force=[0.0, 0.0, 0.2])
    assert r["status"] == "success"
    sim.step(n_steps=5)
    e_after = _content_json(sim.get_energy())["kinetic"]
    assert e_after > e_before, f"KE must grow after force: {e_before} -> {e_after}"

    # raycast straight down from above target → should hit something
    ray = sim.raycast(origin=[0.2, 0.2, 1.0], direction=[0, 0, -1])
    assert ray["status"] == "success"

    # multi_raycast: 3 directions from same origin
    multi = sim.multi_raycast(
        origin=[0.0, 0.0, 1.0],
        directions=[[0, 0, -1], [0, 0.1, -1], [0.1, 0, -1]],
    )
    assert multi["status"] == "success"
    rays_data = _content_json(multi)["rays"]
    assert len(rays_data) == 3

    # jacobian of the target object body
    jac = sim.get_jacobian(body_name="target")
    assert jac["status"] == "success"
    jac_data = _content_json(jac)
    assert jac_data["nv"] > 0
    assert len(jac_data["jacp"]) == 3  # 3 rows (xyz translation jacobian)

    # mass matrix is nv×nv, symmetric-ish, positive diagonal
    mm = sim.get_mass_matrix()
    assert mm["status"] == "success"
    diag = _content_json(mm)["diagonal"]
    assert all(d > 0 for d in diag), "Mass matrix diagonal must be positive"

    # total_mass = sum of body masses, all positive
    tm = sim.get_total_mass()
    assert tm["status"] == "success"
    assert _content_json(tm)["total_mass"] > 0

    # inverse + forward dynamics round-trip (don't compare values, just smoke)
    for m in (
        "inverse_dynamics",
        "forward_kinematics",
        "get_contacts",
        "get_contact_forces",
        "get_body_state",
        "get_sensor_data",
    ):
        result = getattr(sim, m)(body_name="target") if m == "get_body_state" else getattr(sim, m)()
        assert result["status"] == "success", f"{m}: {result}"

    # set_body_properties - bump mass, re-read total_mass
    tm_before = _content_json(sim.get_total_mass())["total_mass"]
    r = sim.set_body_properties(body_name="target", mass=0.5)
    assert r["status"] == "success"
    tm_after = _content_json(sim.get_total_mass())["total_mass"]
    assert abs(tm_after - tm_before) > 1e-6, "Mass change must propagate"

    # set_geom_properties - tweak colour, verify no crash
    r = sim.set_geom_properties(geom_name="target_geom", color=[0.5, 0.5, 0.5, 1.0])
    assert r["status"] == "success"

    # set_joint_velocities - non-zero velocity on the first arm joint
    joints = sim.robot_joint_names("arm")
    r = sim.set_joint_velocities(velocities={joints[0]: 0.3})
    assert r["status"] == "success"
    sim.step(n_steps=5)


# J3 · SNAPSHOT - save_state → perturb → load_state → bit-exact rollback


def test_j3_snapshot_save_load_round_trip(sim):
    """State snapshot must restore qpos exactly. Physics is deterministic."""
    sim.add_object("cube", shape="box", size=[0.025] * 3, position=[0.15, 0.15, 0.05])
    sim.step(n_steps=30)

    qpos_pre = sim._world._data.qpos.copy()

    r = sim.save_state(name="pristine")
    assert r["status"] == "success"

    # Perturb aggressively
    sim.apply_force(body_name="cube", force=[2.0, -1.0, 5.0])
    sim.step(n_steps=50)
    qpos_mid = sim._world._data.qpos.copy()
    assert not np.allclose(qpos_mid, qpos_pre, atol=1e-3), "perturbation must move qpos"

    # Rollback
    r = sim.load_state(name="pristine")
    assert r["status"] == "success"
    qpos_restored = sim._world._data.qpos.copy()
    assert np.allclose(qpos_restored, qpos_pre, atol=1e-9), "snapshot must be bit-exact"


# J4 · POLICY ROLLOUT - mock policy drives the arm, qpos + sim_time advance


def test_j4_policy_mock_rollout_moves_arm(sim, mock_policy):
    """MockPolicy + run_policy(policy_object=...) → real qpos delta.

    Guards the bug we hit earlier: for SmolVLA we *must* build the policy
    before ``run_policy`` so recording doesn't capture a frozen arm during
    weight load. Here we use MockPolicy, but the pre-built path is identical.
    """
    qpos_pre = sim._world._data.qpos.copy()
    t_pre = sim._world.sim_time

    r = sim.run_policy(
        robot_name="arm",
        policy_object=mock_policy,
        duration=0.5,
        control_frequency=30.0,
    )
    assert r["status"] == "success", r
    assert "Policy complete" in _content_texts(r)

    qpos_post = sim._world._data.qpos.copy()
    t_post = sim._world.sim_time

    assert t_post > t_pre, "sim_time must advance"
    delta = float(np.abs(qpos_post - qpos_pre).sum())
    assert delta > 1e-3, f"mock policy must move the arm (Δ={delta})"


# J5 · REPLAY - feed a synthetic "dataset" through PolicyRunner.replay


def test_j5_replay_applies_recorded_actions_to_arm(sim, monkeypatch):
    """PolicyRunner.replay() consumes a dataset frame-by-frame.

    We synthesise the smallest dataset shape that ``replay`` understands:
    a sliceable object yielding ``{"action": [...]}`` per index, plus
    module-level ``load_lerobot_episode(...)`` returning ``(ds, start, length)``.

    This lets us test the full replay loop without a ~GB HF download.
    """
    joints = sim.robot_joint_names("arm")
    n_frames = 30
    # 0 → 0.3 rad sweep on first joint, others flat
    actions = np.zeros((n_frames, len(joints)), dtype=np.float32)
    actions[:, 0] = np.linspace(0.0, 0.3, n_frames)

    class FakeEpisode:
        fps = 30

        def __len__(self):
            return n_frames

        def __getitem__(self, idx):
            return {"action": actions[idx]}

    episode = FakeEpisode()

    def fake_loader(repo_id, episode_idx, root):
        assert repo_id == "synthetic/pr85_replay"
        return episode, 0, n_frames

    # Monkey-patch the module-level loader that replay() calls
    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", fake_loader, raising=False)

    from strands_robots.simulation.policy_runner import PolicyRunner

    qpos_pre = sim._world._data.qpos.copy()
    r = PolicyRunner(sim).replay(
        repo_id="synthetic/pr85_replay",
        robot_name="arm",
        speed=10.0,  # faster than real-time
    )
    assert r["status"] == "success", r
    data = _content_json(r)
    assert data["frames_applied"] == n_frames
    assert data["episode"] == 0
    assert data["robot_name"] == "arm"

    qpos_post = sim._world._data.qpos.copy()
    assert np.abs(qpos_post - qpos_pre).sum() > 1e-3, "replay must move the arm"


# J6 · EVALUATE - multi-episode eval with a string success_fn


def test_j6_evaluate_multi_episode_contact_success(sim, mock_policy):
    """PolicyRunner.evaluate(n_episodes=2, success_fn="contact") must run
    clean, return per-episode results, and expose a numeric success_rate.

    Covers the string-dispatch branch in ``_resolve_success_fn`` that
    previously had 0% coverage.
    """
    # Drop a cube that will collide with the arm - gives contact a chance
    sim.add_object("hit_me", shape="box", size=[0.03] * 3, position=[0.1, 0.2, 0.03])

    from strands_robots.simulation.policy_runner import PolicyRunner

    r = PolicyRunner(sim).evaluate(
        robot_name="arm",
        policy=mock_policy,
        instruction="wiggle",
        n_episodes=2,
        max_steps=30,
        success_fn="contact",
    )
    assert r["status"] == "success", r
    data = _content_json(r)
    assert 0.0 <= data["success_rate"] <= 1.0
    assert data["n_episodes"] == 2
    assert len(data["episodes"]) == 2

    # unknown string should be a clean error - NOT a raise
    bad = PolicyRunner(sim).evaluate(
        robot_name="arm",
        policy=mock_policy,
        n_episodes=1,
        success_fn="does_not_exist",
    )
    assert bad["status"] == "error"


# J7 · MULTI-CAM RECORDING - background recorder concurrent with policy


def test_j7_multicam_recording_concurrent_with_policy(sim, mock_policy, tmp_path):
    """start_cameras_recording → run_policy → stop_cameras_recording,
    one MP4 per camera, non-zero size, no recorder errors.

    Guards the recent 4-camera recorder bug: the background thread fills
    ndarray buffers, the main thread flushes them to MP4 on stop - this
    pattern was introduced to avoid ffmpeg pipe races under concurrent load.
    """
    sim.add_camera("overhead", position=[0, 0, 0.7], target=[0, 0, 0.1])

    r = sim.start_cameras_recording(
        cameras=["cam_front", "overhead"],
        output_dir=str(tmp_path),
        fps=20,
        width=64,
        height=48,
        name="j7",
    )
    assert r["status"] == "success", r

    # mid-recording status must not lie
    status = sim.get_cameras_recording_status()
    assert status["status"] == "success"
    assert "🟢" in _content_texts(status)

    rollout = sim.run_policy(
        robot_name="arm",
        policy_object=mock_policy,
        duration=0.5,
        control_frequency=20.0,
    )
    assert rollout["status"] == "success"

    stop = sim.stop_cameras_recording()
    assert stop["status"] == "success"
    data = _content_json(stop)
    assert data["recording"] == "j7"
    assert len(data["artifacts"]) == 2

    for artifact in data["artifacts"]:
        assert artifact["frames"] > 0, f"no frames captured for {artifact['camera']}"
        assert artifact["errors"] == 0, f"recorder errors on {artifact['camera']}"
        assert Path(artifact["path"]).exists()
        assert Path(artifact["path"]).stat().st_size > 0

    # Post-stop: status is idle, and double-stop is an idempotent no-op.
    # stop_cameras_recording is documented idempotent ("already stopped is a
    # success, not an error"), so a second stop returns success with an
    # informational "was not recording" message rather than erroring.
    status = sim.get_cameras_recording_status()
    assert "⚪" in _content_texts(status)
    double_stop = sim.stop_cameras_recording()
    assert double_stop["status"] == "success"


# J8 · SINGLE-CAMERA RUN_POLICY VIDEO - the path that used to silently fail


def test_j8_run_policy_video_writes_mp4(sim, mock_policy, tmp_path):
    """run_policy(video={...}) must produce a playable MP4.

    This was silently broken for the life of the PR: the recording loop
    used ``frame.get("image")`` on the top-level render result, but images
    are nested inside content blocks → every rollout wrote zero frames
    and crashed on ``os.path.getsize``. Fixed in ``_extract_frame_ndarray``.
    """
    video = tmp_path / "run_policy.mp4"
    r = sim.run_policy(
        robot_name="arm",
        policy_object=mock_policy,
        duration=0.5,
        control_frequency=30.0,
        video={"path": str(video), "fps": 30, "camera": "cam_front", "width": 64, "height": 48},
    )
    assert r["status"] == "success", r
    assert video.exists()
    assert video.stat().st_size > 0

    text = _content_texts(r)
    assert "🎬 Video:" in text
    assert "frames" in text


# J9 · AGENTIC DISPATCH - tool-schema path with real field remapping


def test_j9_agent_dispatch_routes_actions_through_tool_spec(sim):
    """Verify the ``_dispatch_action`` path (what a Strands agent hits).

    Validates:
      * ``list_robots`` action → maps via ``_ALIASES`` to ``list_robots_info``
      * ``render`` action → returns image content block
      * ``step`` action → advances sim_time
      * unknown action → clean error

    This is the exact path an ``Agent(tools=[sim])`` invocation takes.
    """
    # list_robots action → aliased to list_robots_info() (rich dict output)
    r = sim._dispatch_action("list_robots", {"action": "list_robots"})
    assert r["status"] == "success"
    assert "arm" in _content_texts(r)

    # render action → content blocks with an image
    r = sim._dispatch_action(
        "render",
        {"action": "render", "camera_name": "cam_front", "width": 48, "height": 32},
    )
    assert r["status"] == "success"
    assert _n_images(r) == 1

    # step action → sim_time advances
    t_pre = sim._world.sim_time
    r = sim._dispatch_action("step", {"action": "step", "n_steps": 5})
    assert r["status"] == "success"
    assert sim._world.sim_time > t_pre

    # unknown action → error, no raise
    r = sim._dispatch_action("nonexistent", {"action": "nonexistent"})
    assert r["status"] == "error"
    assert "Unknown action" in _content_texts(r)


# J10 · ERROR GRAMMAR - empty sim, every public method, no raises


def test_j10_empty_sim_methods_never_raise():
    """Every public method on an un-initialised Simulation returns a clean
    error dict rather than raising. This is the *API contract* for agent
    tools: the LLM-facing method must never bubble an exception.
    """
    from strands_robots.simulation import Simulation

    s = Simulation()  # no create_world

    methods = [
        ("get_features", ()),
        ("list_objects", ()),
        ("get_state", ()),
        ("inverse_dynamics", ()),
        ("forward_kinematics", ()),
        ("get_energy", ()),
        ("get_total_mass", ()),
        ("get_contacts", ()),
        ("get_contact_forces", ()),
        ("get_sensor_data", ()),
        ("get_mass_matrix", ()),
        ("save_state", ("snap",)),
        ("load_state", ("snap",)),
        ("set_joint_positions", ({"j": 0.0},)),
        ("set_joint_velocities", ({"j": 0.0},)),
        ("get_body_state", ("nobody",)),
        ("apply_force", ("nobody",)),
        ("raycast", ([0, 0, 1], [0, 0, -1])),
        ("get_cameras_recording_status", ()),
        ("render", ()),
        ("render_depth", ()),
        ("render_all", ()),
        ("remove_robot", ("ghost",)),
        ("remove_object", ("ghost",)),
        ("remove_camera", ("ghost",)),
        ("stop_policy", ("ghost",)),
        ("stop_cameras_recording", ()),
        ("stop_recording", ()),
        ("get_recording_status", ()),
    ]

    for name, args in methods:
        method = getattr(s, name)
        result = method(*args)
        assert isinstance(result, dict), f"{name} returned {type(result).__name__}"
        assert result.get("status") in ("success", "error"), f"{name}: {result}"
        # On an empty sim, all state-observing calls should be 'error'
        # Most state-observing methods must error on an empty sim. Status queries
        # that have a meaningful "idle" response (camera recording, regular
        # recording) legitimately return success with an informational message.
        # Idempotent stop/status queries legitimately return an informational
        # success on an empty sim rather than erroring (e.g. "was not recording").
        STATUS_QUERIES_OK_ON_EMPTY = {
            "get_cameras_recording_status",
            "stop_cameras_recording",
            "stop_recording",
            "get_recording_status",
        }
        if (
            name.startswith(
                (
                    "get_",
                    "list_",
                    "render",
                    "save_",
                    "load_",
                    "remove_",
                    "stop_",
                    "apply_",
                    "raycast",
                    "inverse_",
                    "forward_",
                    "set_",
                )
            )
            and name not in STATUS_QUERIES_OK_ON_EMPTY
        ):
            assert result["status"] == "error", f"{name} on empty sim should error, got: {result}"
            txt = _content_texts(result)
            # Every error message conveys failure: an explicit "error", a
            # negation ("No "/"Not "), or a "not found" / "no ..." phrasing.
            txt_l = txt.lower()
            assert "error" in txt_l or "no " in txt_l or "not " in txt_l, f"{name}: {txt!r}"

    s.destroy()


# J11 · LEROBOT DATASET RECORDING - start_recording (episode write round-trip)


def test_j11_lerobot_dataset_recording_round_trip(sim, mock_policy, tmp_path):
    """start_recording → run_policy → stop_recording must write a LeRobotDataset.

    Covers ``RecordingMixin.start_recording`` (14 uncovered lines) and the
    on_frame hook path in ``_make_run_policy_hook`` that appends episode
    frames to the dataset while the policy steps.

    Uses a local root under ``tmp_path`` so we never touch the HF cache
    and the test is fully self-contained + re-runnable.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    rec = sim.start_recording(
        repo_id="local/pr85_j11",
        task="test_j11",
        fps=20,
        root=str(tmp_path),
        overwrite=True,
    )
    assert rec["status"] == "success", rec
    assert "Recording to LeRobotDataset" in _content_texts(rec)

    status_during = sim.get_recording_status()
    assert status_during["status"] == "success"

    r = sim.run_policy(
        robot_name="arm",
        policy_object=mock_policy,
        duration=0.3,
        control_frequency=20.0,
    )
    assert r["status"] == "success"

    stop = sim.stop_recording()
    assert stop["status"] == "success", stop

    # LeRobot datasets emit a parquet per episode + metadata
    written = list(tmp_path.rglob("*"))
    parquets = [p for p in written if p.suffix == ".parquet"]
    jsons = [p for p in written if p.suffix in (".json", ".jsonl")]
    assert parquets or jsons, f"no dataset files written to {tmp_path}: {written[:10]}"
