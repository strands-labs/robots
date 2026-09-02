# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A dataset's declared frame rate must be the rate its frames were captured at.

``start_recording(fps=...)`` fixes the rate written into the LeRobotDataset
metadata, and LeRobot derives every timestamp from it positionally
(``timestamp = frame_index / fps``). The dataset recorder is driven once per
control step with **no decimation**, so the rate frames are really captured at
is the rollout's ``control_frequency`` - a differing ``fps`` cannot be honored,
only mislabelled.

The two library defaults were exactly such a pair (``fps=30`` against
``control_frequency=50.0``), which is what the documented record-then-rollout
sequence in ``docs/recording.md`` used. Measured on a position-servo arm before
the guard::

    fps=30 cf=50.0 -> captured 0.0200s/frame, timestamped 0.0333s/frame (1.667x)
    fps=50 cf=50.0 -> captured 0.0200s/frame, timestamped 0.0200s/frame (1.000x)

with ``start_recording``, the rollout and ``stop_recording`` all reporting
``status="success"`` and no log line. The distortion is the control period a
policy trains on, and ``replay_episode`` derives its per-frame physics budget
from the dataset rate on the invariant that "the recorded control frequency IS
the dataset fps" - so the same episode also replays at the wrong speed
(round-tripped to 0.0000 rad at matching rates, 0.0317 rad at the defaults).

The disagreement has three orderings and all three are refused. A rollout
started against an open recording is refused at every rollout entry point; a
recording opened against a rollout already in flight is refused by
``start_recording``. ``start_policy`` makes the second ordering reachable by
design - it submits the rollout to an executor and returns while it continues -
and on the same colliding defaults it saved 81 frames captured 0.0200s apart and
declared them 0.0333s apart: a 2.6667s episode for a 1.62s capture, with
``start_policy``, ``start_recording`` and ``stop_recording`` all reporting
success. The third ordering is a single call that supplies both rates and opens
the recording itself - the ``run_policy`` tool - where neither rate can be read
off live state because nothing is open yet; it is refused by
``requested_rate_mismatch_reason`` and covered in
``tests/tools/test_run_policy_rate_agreement_preflight.py``.

Refused rather than warned, matching the sibling rate guard in the same module:
``_verify_resume_schema`` already refuses an ``fps`` that disagrees with the
dataset on disk. The refusal lands before any frame is written - and, in the
inverse ordering, before any dataset is created - so a caller loses nothing.
"""

from __future__ import annotations

import ast
import contextlib
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402
from strands_robots.simulation.recording import (  # noqa: E402
    dataset_rate_mismatch_error,
    dataset_rate_mismatch_reason,
    dataset_recording_option_error,
    rate_mismatch_explanation,
    recorder_dataset_fps,
    requested_rate_mismatch_reason,
    rollout_rate_mismatch_error,
    rollout_rate_mismatch_reason,
)

_ARM = """<mujoco><worldbody><body name="l1">
<joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/>
<body name="l2" pos="0.15 0 0">
<joint name="j2" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/></body></body></worldbody>
<actuator><position name="a1" joint="j1" kp="30" ctrlrange="-1.5 1.5"/>
<position name="a2" joint="j2" kp="30" ctrlrange="-1.5 1.5"/></actuator></mujoco>"""


class _Hold(Policy):
    """State-only policy: no camera renders, so these tests stay fast."""

    def __init__(self, keys: list[str]) -> None:
        super().__init__()
        self._keys = list(keys)

    @property
    def provider_name(self) -> str:
        return "hold"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, keys) -> None:  # noqa: ANN001
        pass

    async def get_actions(self, observation_dict, instruction, **kwargs):  # noqa: ANN001, ANN003
        return [{k: 0.3 for k in self._keys}]


@pytest.fixture
def sim(tmp_path):
    """A two-actuator arm needing no asset download."""
    xml = tmp_path / "arm.xml"
    xml.write_text(_ARM)
    engine = MuJoCoSimEngine(tool_name="rate_guard_sim", mesh=False)
    engine.create_world()
    engine.add_robot(name="arm", urdf_path=str(xml))
    yield engine
    engine.cleanup()


def _record(sim, tmp_path, fps: int, name: str = "ds") -> str:
    """Open a recording at ``fps``; each case gets its OWN root.

    ``start_recording`` RESUMES an existing dataset directory and inherits its
    rate from disk, so sharing one root across cases makes every case report
    the first one's fps - which looks exactly like ``fps`` being ignored.
    """
    root = tmp_path / name
    result = sim.start_recording(repo_id=f"local/{name}", task="hold", fps=fps, root=str(root))
    assert result["status"] == "success"
    return str(root)


def _frames_on_disk(root: str) -> int:
    parquets = [p for p in Path(root).rglob("*.parquet") if "data" in p.parts]
    if not parquets:
        return 0
    pd = pytest.importorskip("pandas")
    return sum(len(pd.read_parquet(p)) for p in parquets)


class TestTheLibraryDefaultsAreRefused:
    """The out-of-the-box pair is the mismatch, so it is the headline case."""

    def test_recording_at_30_then_rolling_at_50_is_refused(self, sim, tmp_path):
        _record(sim, tmp_path, 30)
        result = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), n_steps=10)
        assert result["status"] == "error"

    def test_the_refusal_names_both_rates_and_the_distortion(self, sim, tmp_path):
        _record(sim, tmp_path, 30)
        text = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), n_steps=10)["content"][0]["text"]
        assert "30 fps" in text
        assert "control_frequency=50" in text
        assert "1.667x" in text

    def test_the_refusal_names_both_remedies(self, sim, tmp_path):
        _record(sim, tmp_path, 30)
        text = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), n_steps=10)["content"][0]["text"]
        assert "control_frequency=30" in text
        assert "start_recording(fps=50, overwrite=True)" in text

    def test_no_frame_is_written(self, sim, tmp_path):
        """The refusal precedes capture, so the caller loses no episode."""
        root = _record(sim, tmp_path, 30)
        sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), n_steps=10)
        assert _frames_on_disk(root) == 0


class TestMatchingRatesAreUnaffected:
    def test_a_rollout_at_the_declared_rate_records(self, sim, tmp_path):
        root = _record(sim, tmp_path, 50)
        result = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), control_frequency=50.0, n_steps=10)
        assert result["status"] == "success"
        sim.stop_recording()
        assert _frames_on_disk(root) == 10

    def test_the_episode_duration_is_not_distorted(self, sim, tmp_path):
        """The declared span must equal the captured span - the point of the guard."""
        pd = pytest.importorskip("pandas")
        root = _record(sim, tmp_path, 25)
        sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), control_frequency=25.0, n_steps=10)
        sim.stop_recording()
        parquets = [p for p in Path(root).rglob("*.parquet") if "data" in p.parts]
        stamps = pd.concat([pd.read_parquet(p) for p in parquets])["timestamp"].tolist()
        assert stamps[-1] - stamps[0] == pytest.approx((len(stamps) - 1) / 25.0, abs=1e-6)

    def test_a_rollout_with_no_recording_open_is_never_refused(self, sim):
        result = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), control_frequency=17.0, n_steps=5)
        assert result["status"] == "success"


class TestTheAdvisedRemedyIsUsable:
    """A recommendation that would not work must not be printed."""

    def test_following_the_control_frequency_remedy_records_cleanly(self, sim, tmp_path):
        root = _record(sim, tmp_path, 30)
        refused = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), n_steps=10)
        assert refused["status"] == "error"
        # The message says: pass control_frequency=30. Do exactly that.
        result = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), control_frequency=30.0, n_steps=10)
        assert result["status"] == "success"
        sim.stop_recording()
        assert _frames_on_disk(root) == 10

    def test_a_fractional_capture_rate_offers_only_the_rate_it_can(self, sim, tmp_path):
        """``fps`` must be a whole number, so 33.3 Hz has no re-record remedy."""
        _record(sim, tmp_path, 30)
        text = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), control_frequency=33.3, n_steps=5)[
            "content"
        ][0]["text"]
        assert "control_frequency=30" in text
        assert "start_recording(fps=" not in text


class TestTheGuardHelperIsRobust:
    def test_matching_rates_return_none(self):
        assert dataset_rate_mismatch_error("run_policy", _FakeRecorder(30), 30.0) is None

    def test_float_noise_below_the_tolerance_is_not_a_mismatch(self):
        """A rate carried as a float must not be refused for representation noise."""
        assert dataset_rate_mismatch_error("run_policy", _FakeRecorder(30), 30.0 + 1e-12) is None

    def test_a_dataset_with_no_readable_rate_does_not_block(self):
        """An unexpected LeRobot layout must not refuse a valid rollout."""
        assert dataset_rate_mismatch_error("run_policy", _FakeRecorder(None), 50.0) is None

    def test_a_fractional_dataset_rate_does_not_block(self):
        assert dataset_rate_mismatch_error("run_policy", _FakeRecorder(29.97), 50.0) is None

    def test_the_message_is_prefixed_with_the_calling_method(self):
        err = dataset_rate_mismatch_error("eval_policy", _FakeRecorder(30), 50.0)
        assert err is not None
        assert err["content"][0]["text"].startswith("eval_policy: ")

    @pytest.mark.parametrize("rate", [30, 30.0])
    def test_the_rate_is_read_from_either_lerobot_spelling(self, rate):
        assert recorder_dataset_fps(_FakeRecorder(rate)) == 30
        assert recorder_dataset_fps(_MetaOnlyRecorder(rate)) == 30

    def test_a_boolean_rate_is_not_read_as_one(self):
        assert recorder_dataset_fps(_FakeRecorder(True)) is None
        assert recorder_dataset_fps(_FakeRecorder(np.bool_(True))) is None

    @pytest.mark.parametrize("rate", [np.int64(30), np.int32(30), np.float32(30.0), np.float64(30.0)])
    def test_a_numpy_rate_is_read_rather_than_treated_as_an_unreadable_layout(self, rate):
        """A whole rate is a whole rate whichever scalar type carries it.

        ``numpy.int64`` and ``numpy.float32`` are not ``int``/``float``
        subclasses, so classifying on ``int | float`` reported them as "no
        readable rate" - and the one caller reads that as "do not judge", so the
        mismatch refusal was skipped and the episode was written on a timebase
        that mislabels it. ``numpy.float64`` IS a ``float`` subclass, so the
        narrowing held for one numpy spelling and not its siblings.
        """
        assert recorder_dataset_fps(_FakeRecorder(rate)) == 30
        assert recorder_dataset_fps(_MetaOnlyRecorder(rate)) == 30
        # A built-in ``int``, not the numpy scalar passed through: the annotation
        # says ``int``, and the value is handed to ``1.0 / fps`` and into a reason
        # string. ``== 30`` holds for a numpy scalar too, so the identity is
        # asserted rather than left implied by the equality above.
        assert type(recorder_dataset_fps(_FakeRecorder(rate))) is int

    @pytest.mark.parametrize("rate", [10**400, -(10**400)])
    def test_a_rate_beyond_the_float_range_is_an_unreadable_layout_not_an_exception(self, rate):
        """A rate no recording can be written at is reported, not raised.

        This rate arrives off disk rather than from a caller, so no domain has
        classified it: ``meta/info.json`` is JSON, whose integer literals are
        unbounded, and LeRobot's ``fps`` field is an unenforced dataclass
        annotation - so ``LeRobotDataset`` opens such a dataset without complaint
        and hands the 401-digit ``int`` straight to this reader. Letting the
        conversion raise escaped a function documented to answer ``None`` for a
        layout it cannot read, and the caller reported it as the dataset having
        failed to open, which it had not. Measured through ``start_recording``
        against a recorded dataset whose on-disk ``fps`` was edited to that
        value: ``error: Dataset init failed: int too large to convert to float``,
        naming neither the field nor a remedy, while the fractional and infinite
        rates beside it resumed as the unreadable layouts they are.

        Both signs are asserted because resolving the rate through ``float`` -
        which the typed spelling requires, ``numbers.Real`` carrying no ordering
        against ``int`` - converts before it can test the sign, so a negative rate
        reaches the conversion as well. Only the positive one changes what
        ``start_recording`` reports: LeRobot refuses a negative rate as it opens
        the dataset ("fps must be positive"), so that sign is a raise this reader
        owes its own callers rather than a verdict the surface got wrong. It is
        pinned here, where the contract is, for that reason.
        """
        assert recorder_dataset_fps(_FakeRecorder(rate)) is None
        assert recorder_dataset_fps(_MetaOnlyRecorder(rate)) is None


class TestTheRunnerLayerCarriesItsOwnGuarantee:
    """``PolicyRunner`` is driven directly, with the engine's guard off the path.

    ``docs/policies/lerobot-local.md`` names ``PolicyRunner.run`` beside
    ``run_policy`` as a caller surface, and ``_control_substeps`` already raises
    for a bad ``control_substeps`` on the stated grounds that "the public entry
    points reject such a value ... this raise is the guarantee for callers
    driving ``PolicyRunner`` directly". The recording rate needs the same
    treatment: measured before the guard, a direct rollout at ``fps=30`` /
    ``control_frequency=50`` wrote 20 frames declaring 0.0333s each for a
    capture 0.0200s apart (1.667x) and reported ``status="success"``.
    """

    def _keys(self, sim) -> list[str]:  # noqa: ANN001
        return list(sim.robot_action_keys("arm"))

    def test_direct_run_refuses_a_rate_the_recording_cannot_describe(self, sim, tmp_path):
        root = _record(sim, tmp_path, fps=30)
        with pytest.raises(ValueError) as excinfo:
            PolicyRunner(sim).run(
                "arm",
                _Hold(self._keys(sim)),
                n_steps=20,
                control_frequency=50.0,
                on_frame=sim._make_run_policy_hook("arm", "hold"),
            )
        message = str(excinfo.value)
        assert message.startswith("PolicyRunner.run: ")
        assert "declares 30 fps" in message
        assert "control_frequency=50 Hz" in message
        # Refused before the recorder was touched, so nothing has to be undone.
        assert sim._active_recorder().frame_count == 0
        assert _frames_on_disk(root) == 0

    def test_direct_evaluate_refuses_the_same_disagreement(self, sim, tmp_path):
        """The eval loop writes into the same open recording, so it refuses too."""
        root = _record(sim, tmp_path, fps=30, name="eval_ds")
        with pytest.raises(ValueError) as excinfo:
            PolicyRunner(sim).evaluate(
                "arm",
                _Hold(self._keys(sim)),
                n_episodes=1,
                max_steps=20,
                control_frequency=50.0,
                on_frame=sim._make_run_policy_hook("arm", "hold"),
            )
        assert str(excinfo.value).startswith("PolicyRunner.evaluate: ")
        assert _frames_on_disk(root) == 0

    def test_a_matching_rate_still_records_an_exact_timebase(self, sim, tmp_path):
        """Control: the guard must not cost a correctly configured direct caller."""
        root = _record(sim, tmp_path, fps=50, name="aligned_ds")
        result = PolicyRunner(sim).run(
            "arm",
            _Hold(self._keys(sim)),
            n_steps=20,
            control_frequency=50.0,
            on_frame=sim._make_run_policy_hook("arm", "hold"),
        )
        assert result["status"] == "success"
        assert sim.stop_recording()["status"] == "success"
        assert _frames_on_disk(root) == 20
        pd = pytest.importorskip("pandas")
        parquets = [p for p in Path(root).rglob("*.parquet") if "data" in p.parts]
        stamps = pd.concat([pd.read_parquet(p) for p in parquets])["timestamp"].tolist()
        assert stamps[1] - stamps[0] == pytest.approx(1.0 / 50.0, abs=1e-9)

    def test_a_rollout_with_no_recording_open_is_unaffected(self, sim):
        """Control: the rates are only comparable while a recording is open."""
        result = PolicyRunner(sim).run("arm", _Hold(self._keys(sim)), n_steps=5, control_frequency=50.0)
        assert result["status"] == "success"

    def test_a_sim_without_the_recording_hooks_is_not_probed(self):
        """A backend that cannot record, or a test double, has neither hook."""
        PolicyRunner(object())._reject_recording_rate_mismatch(50.0, "PolicyRunner.run")

    def test_a_sim_reporting_no_active_recorder_is_not_probed(self):
        """``_is_recording`` and ``_active_recorder`` can disagree; trust neither alone."""

        class _Engine:
            def _is_recording(self) -> bool:
                return True

            def _active_recorder(self) -> None:
                return None

        PolicyRunner(_Engine())._reject_recording_rate_mismatch(50.0, "PolicyRunner.run")


@pytest.mark.parametrize(
    ("fps", "control_frequency"),
    [(30, 50.0), (50, 30.0), (30, 30.0), (50, 50.0), (25, 25.0), (None, 50.0), (29.97, 50.0)],
)
def test_the_runner_and_the_engine_refuse_the_same_rates(fps, control_frequency):
    """One rule, two surfaces: the verdict and the reason must not diverge.

    The engine reports through a tool envelope and the runner raises, so only a
    shared reason keeps a caller from being told two different things about the
    same pair of rates.
    """
    recorder = _FakeRecorder(fps)
    envelope = dataset_rate_mismatch_error("run_policy", recorder, control_frequency)
    reason = dataset_rate_mismatch_reason("run_policy", recorder, control_frequency)
    assert (envelope is None) == (reason is None), (
        f"the two surfaces disagree for fps={fps!r} control_frequency={control_frequency!r}"
    )
    if envelope is not None and reason is not None:
        assert envelope["content"][0]["text"] == reason


def test_the_reason_names_the_method_the_caller_actually_called():
    """The remedy must advise changing ``PolicyRunner.run``, not ``run_policy``."""
    reason = dataset_rate_mismatch_reason("PolicyRunner.run", _FakeRecorder(30), 50.0)
    assert reason is not None
    assert reason.startswith("PolicyRunner.run: ")
    assert "pass control_frequency=30 to PolicyRunner.run()" in reason
    assert "run_policy" not in reason


class _Dataset:
    def __init__(self, fps, meta: object | None = None) -> None:  # noqa: ANN001
        self.fps = fps
        self.meta = meta


class _Meta:
    def __init__(self, fps) -> None:  # noqa: ANN001
        self.fps = fps


class _FakeRecorder:
    """Exposes the rate the way ``LeRobotDataset`` does, directly."""

    def __init__(self, fps) -> None:  # noqa: ANN001
        self.dataset = _Dataset(fps)


class _MetaOnlyRecorder:
    """A layout that carries the rate only on the metadata object."""

    def __init__(self, fps) -> None:  # noqa: ANN001
        self.dataset = _Dataset(None, meta=_Meta(fps))


@pytest.fixture
def two_arm_sim(tmp_path):
    """Two independent arms, so two ``start_policy`` rollouts can overlap."""
    xml = tmp_path / "arm.xml"
    xml.write_text(_ARM)
    engine = MuJoCoSimEngine(tool_name="rate_guard_multi", mesh=False)
    engine.create_world()
    engine.add_robot(name="armA", urdf_path=str(xml))
    engine.add_robot(name="armB", urdf_path=str(xml), position=[0.6, 0.0, 0.0])
    yield engine
    engine.cleanup()


@contextlib.contextmanager
def _running_rollout(sim, robot: str, control_frequency: float):
    """A real in-flight ``start_policy`` rollout on ``robot``, stopped on exit.

    The horizon is long enough that the rollout cannot finish mid-test and be
    pruned, which would otherwise look like the guard failing to fire. Waits for
    the worker to have installed its per-frame hook before yielding: until then
    the hook has not set ``policy_running``, so a ``stop_policy`` on exit would be
    overwritten and the rollout would outlive the test.
    """
    keys = sim.robot_action_keys(robot)
    result = sim.start_policy(
        robot_name=robot,
        policy_object=_Hold(keys),
        duration=60.0,
        control_frequency=control_frequency,
    )
    assert result["status"] == "success", result
    handle = sim._world.robots[robot]
    deadline = time.monotonic() + 20.0
    while not handle.policy_running:
        if time.monotonic() > deadline:
            pytest.fail(f"the rollout on {robot!r} never started")
        time.sleep(0.005)
    try:
        yield
    finally:
        sim.stop_policy(robot_name=robot)
        future = sim._policy_threads.get(robot)
        if future is not None:
            # Joined without suppressing: a rollout that outlives its stop would
            # otherwise leak into the next case as a spurious "already running".
            future.result(timeout=30.0)


class TestARecordingOpenedAgainstARunningRolloutIsRefused:
    """The inverse ordering: ``start_policy`` first, then ``start_recording``."""

    def test_the_library_defaults_are_refused(self, sim, tmp_path):
        """``control_frequency=50.0`` running, ``fps=30`` requested - the default pair."""
        with _running_rollout(sim, "arm", 50.0):
            result = sim.start_recording(repo_id="local/inverse", task="hold", fps=30, root=str(tmp_path / "inv"))
        assert result["status"] == "error"

    def test_the_refusal_names_the_rollout_both_rates_and_the_distortion(self, sim, tmp_path):
        with _running_rollout(sim, "arm", 50.0):
            text = sim.start_recording(repo_id="local/inverse", task="hold", fps=30, root=str(tmp_path / "inv"))[
                "content"
            ][0]["text"]
        assert "'arm' at 50 Hz" in text
        assert "30 fps" in text
        assert "1.667x" in text

    def test_the_refusal_names_both_remedies(self, sim, tmp_path):
        with _running_rollout(sim, "arm", 50.0):
            text = sim.start_recording(repo_id="local/inverse", task="hold", fps=30, root=str(tmp_path / "inv"))[
                "content"
            ][0]["text"]
        assert "start_recording(fps=50)" in text
        assert "stop_policy(robot_name='arm')" in text
        assert "control_frequency=30" in text

    def test_no_dataset_is_created(self, sim, tmp_path):
        """The refusal precedes dataset creation, so nothing is left on disk."""
        root = tmp_path / "inv"
        with _running_rollout(sim, "arm", 50.0):
            sim.start_recording(repo_id="local/inverse", task="hold", fps=30, root=str(root))
        assert not root.exists() or not any(root.iterdir())

    def test_the_recording_session_is_not_left_open(self, sim, tmp_path):
        """A refused open must not flip the engine into a recording state."""
        with _running_rollout(sim, "arm", 50.0):
            sim.start_recording(repo_id="local/inverse", task="hold", fps=30, root=str(tmp_path / "inv"))
            assert sim._is_recording() is False
            assert sim._active_recorder() is None

    def test_a_matching_rate_is_accepted_while_the_rollout_runs(self, sim, tmp_path):
        """The agreeing case is untouched - the guard refuses disagreement only."""
        with _running_rollout(sim, "arm", 50.0):
            result = sim.start_recording(repo_id="local/agree", task="hold", fps=50, root=str(tmp_path / "agree"))
            assert result["status"] == "success", result
            sim.stop_recording()

    def test_an_unusable_fps_is_still_reported_as_the_parameter_error(self, sim, tmp_path):
        """Name-and-value guards keep priority: ``fps`` itself is the complaint."""
        with _running_rollout(sim, "arm", 50.0):
            text = sim.start_recording(repo_id="local/bad", task="hold", fps=2.7, root=str(tmp_path / "bad"))[
                "content"
            ][0]["text"]
        assert "fps" in text
        assert "already running" not in text


class TestTheAdvisedRemedyIsUsableInTheInverseOrdering:
    """Every option the refusal names must actually work when followed."""

    def test_recording_at_the_rollouts_rate_records_cleanly(self, sim, tmp_path):
        """The message says: start_recording(fps=50). Do exactly that."""
        with _running_rollout(sim, "arm", 50.0):
            refused = sim.start_recording(repo_id="local/r1", task="hold", fps=30, root=str(tmp_path / "r1"))
            assert refused["status"] == "error"
            assert "start_recording(fps=50)" in refused["content"][0]["text"]
            accepted = sim.start_recording(repo_id="local/r1", task="hold", fps=50, root=str(tmp_path / "r1"))
            assert accepted["status"] == "success", accepted
            sim.stop_recording()

    def test_restarting_the_rollout_at_the_recordings_rate_records_cleanly(self, sim, tmp_path):
        """The message says: stop_policy, then start_policy(control_frequency=30)."""
        with _running_rollout(sim, "arm", 50.0):
            refused = sim.start_recording(repo_id="local/r2", task="hold", fps=30, root=str(tmp_path / "r2"))
            assert refused["status"] == "error"
        # The context already ran stop_policy(robot_name='arm'), the first half of
        # the remedy; now restart at the recording's rate as advised.
        with _running_rollout(sim, "arm", 30.0):
            accepted = sim.start_recording(repo_id="local/r2", task="hold", fps=30, root=str(tmp_path / "r2"))
            assert accepted["status"] == "success", accepted
            sim.stop_recording()


class TestConcurrentRolloutsAtDifferentRatesAreRefusedOutright:
    """Interleaved frames on one declared rate cannot describe two capture rates."""

    def test_two_rates_are_refused_even_when_fps_matches_one_of_them(self, two_arm_sim, tmp_path):
        with _running_rollout(two_arm_sim, "armA", 50.0), _running_rollout(two_arm_sim, "armB", 25.0):
            result = two_arm_sim.start_recording(
                repo_id="local/multi", task="hold", fps=50, root=str(tmp_path / "multi")
            )
        assert result["status"] == "error"

    def test_the_refusal_names_every_rollout_and_its_rate(self, two_arm_sim, tmp_path):
        with _running_rollout(two_arm_sim, "armA", 50.0), _running_rollout(two_arm_sim, "armB", 25.0):
            text = two_arm_sim.start_recording(
                repo_id="local/multi", task="hold", fps=30, root=str(tmp_path / "multi")
            )["content"][0]["text"]
        assert "'armA' at 50 Hz" in text
        assert "'armB' at 25 Hz" in text
        assert "2 different capture rates" in text

    def test_two_rollouts_at_one_shared_rate_may_record_at_that_rate(self, two_arm_sim, tmp_path):
        with _running_rollout(two_arm_sim, "armA", 40.0), _running_rollout(two_arm_sim, "armB", 40.0):
            result = two_arm_sim.start_recording(
                repo_id="local/shared", task="hold", fps=40, root=str(tmp_path / "shared")
            )
            assert result["status"] == "success", result
            two_arm_sim.stop_recording()


class TestOnlyLiveRolloutsCanBlockARecording:
    """A finished rollout must not keep refusing a rate it no longer captures."""

    def test_no_rollout_running_never_refuses(self, sim, tmp_path):
        result = sim.start_recording(repo_id="local/idle", task="hold", fps=30, root=str(tmp_path / "idle"))
        assert result["status"] == "success"
        sim.stop_recording()

    def test_a_stopped_rollout_stops_blocking(self, sim, tmp_path):
        with _running_rollout(sim, "arm", 50.0):
            assert sim._active_rollout_rates() == {"arm": 50.0}
        # The context stopped and joined the rollout; its rate must be gone.
        assert sim._active_rollout_rates() == {}
        result = sim.start_recording(repo_id="local/after", task="hold", fps=30, root=str(tmp_path / "after"))
        assert result["status"] == "success", result
        sim.stop_recording()

    def test_the_rate_table_is_swept_when_a_robot_is_removed(self, sim):
        """``remove_robot`` drops the Future directly; the rate must follow it."""
        with _running_rollout(sim, "arm", 50.0):
            assert sim._active_rollout_rates() == {"arm": 50.0}
            assert sim.remove_robot("arm")["status"] == "success"
            assert sim._active_rollout_rates() == {}
            assert sim._policy_rates == {}


class TestTheRolloutRateGuardHelperIsRobust:
    def test_no_running_rollout_returns_none(self):
        assert rollout_rate_mismatch_reason("start_recording", 30, {}) is None

    def test_an_agreeing_rate_returns_none(self):
        assert rollout_rate_mismatch_reason("start_recording", 50, {"arm": 50.0}) is None

    def test_float_noise_below_the_tolerance_is_not_a_mismatch(self):
        assert rollout_rate_mismatch_reason("start_recording", 50, {"arm": 50.0 + 1e-12}) is None

    @pytest.mark.parametrize("fps", [2.7, True, "30", None, 0, -5, float("nan"), float("inf")])
    def test_an_fps_outside_the_writable_domain_is_left_to_its_own_guard(self, fps):
        """Reporting it here would name a rate disagreement instead of the parameter."""
        assert rollout_rate_mismatch_reason("start_recording", fps, {"arm": 50.0}) is None

    def test_an_integral_float_fps_is_read_as_that_whole_rate(self):
        assert rollout_rate_mismatch_reason("start_recording", 50.0, {"arm": 50.0}) is None
        assert rollout_rate_mismatch_reason("start_recording", 30.0, {"arm": 50.0}) is not None

    @pytest.mark.parametrize("fps", [np.int64(30), np.int32(30), np.float32(30.0), np.float64(30.0)])
    def test_a_numpy_fps_the_domain_accepts_is_judged_and_not_passed_through(self, fps):
        """Declining to judge an in-domain rate is the distortion, not a nicety.

        ``positive_whole_number_error`` - the domain ``start_recording`` runs
        ``fps`` through before this guard is asked - accepts every one of these
        (pinned by ``tests/simulation/test_dataset_recording_fps_contract.py``).
        So an ``int | float`` narrowing here refuses nothing and reports nothing:
        ``start_recording`` returns ``status="success"`` and the episode declares
        30 fps for frames captured at 50 Hz.
        """
        assert dataset_recording_option_error("start_recording", fps) is None, "precondition: in domain"
        assert rollout_rate_mismatch_reason("start_recording", fps, {"arm": 50.0}) is not None
        assert rollout_rate_mismatch_reason("start_recording", fps, {"arm": 30.0}) is None, "agreeing rate"

    def test_a_fractional_capture_rate_offers_only_the_remedy_it_can(self):
        """``fps`` must be whole, so 33.3 Hz has no rate to advise recording at."""
        reason = rollout_rate_mismatch_reason("start_recording", 30, {"arm": 33.3})
        assert reason is not None
        assert "record at the rollout's rate" not in reason
        assert "stop_policy(robot_name='arm')" in reason

    def test_the_message_is_prefixed_with_the_calling_method(self):
        reason = rollout_rate_mismatch_reason("start_recording", 30, {"arm": 50.0})
        assert reason is not None
        assert reason.startswith("start_recording: ")

    def test_the_envelope_carries_the_reason_verbatim(self):
        envelope = rollout_rate_mismatch_error("start_recording", 30, {"arm": 50.0})
        reason = rollout_rate_mismatch_reason("start_recording", 30, {"arm": 50.0})
        assert envelope is not None
        assert envelope["content"][0]["text"] == reason
        assert envelope["status"] == "error"

    def test_a_backend_with_no_async_rollout_reports_no_rates(self):
        """The base hook is what makes one call site per backend safe."""
        from strands_robots.simulation.base import SimEngine

        assert SimEngine._active_rollout_rates(object()) == {}  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fps", "rate"),
    [
        (30, 50.0),
        (50, 30.0),
        (30, 30.0),
        (50, 50.0),
        (25, 25.0),
        (30, 33.3),
        (60, 50.0),
        # The same pairs with ``fps`` carried by a numpy scalar. An ``fps`` read
        # out of a config or computed from array shape arrives spelled this way,
        # the fps domain accepts all of them, and two of the three orderings
        # classified on ``int | float`` - so the orderings disagreed about a pair
        # every one of them accepts, which is exactly what this test forbids.
        (np.int64(30), 50.0),
        (np.int32(30), 50.0),
        (np.float32(30.0), 50.0),
        (np.float64(30.0), 50.0),
        (np.int64(50), 50.0),
    ],
)
def test_every_ordering_reaches_the_same_verdict_and_explanation(fps, rate):
    """One disagreement, so its orderings may not disagree about it.

    Whichever call came first - and whether either had happened yet - the frames
    and the timestamps come from the same two rates. So a caller who reverses the
    order of two calls, or supplies both rates at once, must not be told the pair
    is acceptable in one direction and not the other, nor be given a different
    account of the distortion in each.
    """
    verdicts = {
        "rollout against an open recording": dataset_rate_mismatch_reason("run_policy", _FakeRecorder(fps), rate),
        "recording against a running rollout": rollout_rate_mismatch_reason("start_recording", fps, {"arm": rate}),
        "both rates in one call": requested_rate_mismatch_reason("run_policy", fps, rate),
    }
    refused = {name for name, reason in verdicts.items() if reason is not None}
    assert refused in (set(), set(verdicts)), (
        f"the orderings disagree for fps={fps!r} capture rate={rate!r}: only {sorted(refused)} refused it"
    )
    shared = rate_mismatch_explanation(fps, rate)
    for name, reason in verdicts.items():
        if reason is not None:
            assert shared in reason, f"{name} gives its own account of the distortion"


_START_RECORDING_BACKENDS = (
    "strands_robots/simulation/mujoco/recording.py",
    "strands_robots/simulation/isaac/recording.py",
    "strands_robots/simulation/newton/recording.py",
)


@pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
def test_every_backend_start_recording_checks_the_running_rollout_rate(module):
    """``start_recording`` is per backend, so each copy must reach the shared guard.

    Structural rather than behavioural for a measured reason: this refusal is
    unreachable on the Isaac and Newton backends. The guard compares against
    :meth:`~strands_robots.simulation.base.SimEngine._active_rollout_rates`,
    which only the MuJoCo backend overrides, so both of the others inherit an
    empty mapping and it returns ``None`` for every ``fps``. Pinning the call
    is what stops a backend that grows an asynchronous rollout later from
    inheriting a ``start_recording`` that silently skipped the check.

    The three sibling refusals that *are* reachable there - ``fps``, the
    posture flags and ``cameras`` - are driven per backend in
    ``test_recording_preflight_refusals_across_backends.py``, which also pins the
    inherited empty mapping this reason rests on.
    """
    assert "_validate_recording_start_rate" in _self_calls(module, "start_recording"), (
        f"{module}::start_recording does not check the running rollout rate"
    )


class TestTheAsyncEntryPointRefusesBeforeItReportsStarted:
    """``start_policy`` returns while its rollout continues, so it must refuse synchronously.

    Seven surfaces check the recording rate. Five had returned the refusal at
    least once; the two that exist only on the MuJoCo backend - ``start_policy``
    and ``run_multi_policy`` - had not, and were pinned only by the structural
    sweep below on the stated grounds that a driver may need "a checkpoint, a
    benchmark registration or a live background thread to reach". Neither of
    these needs one: the rate guard sits above ``self._executor.submit``, so the
    refusal is returned on the caller's own thread with no worker in existence.

    That placement is the whole point of the guard here, in ``start_policy``'s
    own words - "a refusal after submit would report 'started' to a caller whose
    rollout cannot be recorded correctly". A structural pin cannot see it: it
    proves the guard is *called*, never that its refusal is *returned*, so
    discarding the ``return`` while keeping the call satisfies it.
    """

    def _start(self, sim, control_frequency: float):  # noqa: ANN001, ANN202
        """Start a long rollout on ``arm``; the horizon outlives any single case."""
        return sim.start_policy(
            robot_name="arm",
            policy_object=_Hold(list(sim.robot_action_keys("arm"))),
            duration=60.0,
            control_frequency=control_frequency,
        )

    def test_the_library_defaults_are_refused(self, sim, tmp_path):
        _record(sim, tmp_path, fps=30)
        assert self._start(sim, 50.0)["status"] == "error"

    def test_the_refusal_names_both_rates_and_the_distortion(self, sim, tmp_path):
        _record(sim, tmp_path, fps=30)
        text = self._start(sim, 50.0)["content"][0]["text"]
        assert text.startswith("start_policy: ")
        assert "declares 30 fps" in text
        assert "control_frequency=50 Hz" in text
        assert "1.667x" in text

    def test_the_refusal_names_both_remedies(self, sim, tmp_path):
        _record(sim, tmp_path, fps=30)
        text = self._start(sim, 50.0)["content"][0]["text"]
        assert "control_frequency=30" in text
        assert "start_recording(fps=50, overwrite=True)" in text

    def test_the_envelope_is_the_shared_helper_verbatim(self, sim, tmp_path):
        """One rule, one wording: the entry point adds nothing but its own name."""
        _record(sim, tmp_path, fps=30)
        assert self._start(sim, 50.0) == dataset_rate_mismatch_error("start_policy", sim._active_recorder(), 50.0)

    def test_no_rollout_is_started(self, sim, tmp_path):
        """The refusal must not be a false "started": no worker, no running flag."""
        _record(sim, tmp_path, fps=30)
        assert self._start(sim, 50.0)["status"] == "error"
        assert sim._world.robots["arm"].policy_running is False
        assert sim._policy_threads == {}

    def test_no_frame_is_written(self, sim, tmp_path):
        root = _record(sim, tmp_path, fps=30)
        self._start(sim, 50.0)
        assert sim._active_recorder().frame_count == 0
        assert _frames_on_disk(root) == 0

    def test_a_matching_rate_still_starts_the_rollout(self, sim, tmp_path):
        """Control: an aligned rate must still reach the executor and run."""
        _record(sim, tmp_path, fps=50, name="aligned_async")
        with _running_rollout(sim, "arm", 50.0):
            assert sim._world.robots["arm"].policy_running is True

    def test_a_rollout_with_no_recording_open_is_never_refused(self, sim):
        """Control: the two rates are only comparable while a recording is open."""
        with _running_rollout(sim, "arm", 50.0):
            assert sim._world.robots["arm"].policy_running is True


class TestTheMultiRobotEntryPointRefusesTheSameDisagreement:
    """``run_multi_policy`` writes one merged frame per step into the same recording.

    It is the recommended path for capturing two robots into a single dataset, so
    a rate it cannot honor mislabels the merged episode exactly as the
    single-robot path does - and it reaches the guard with nothing more than two
    robots, no checkpoint and no thread.
    """

    def _policies(self, sim):  # noqa: ANN001, ANN202
        return {name: _Hold(list(sim.robot_action_keys(name))) for name in ("armA", "armB")}

    def _run(self, sim, control_frequency: float):  # noqa: ANN001, ANN202
        return sim.run_multi_policy(
            policies=self._policies(sim),
            n_steps=10,
            control_frequency=control_frequency,
        )

    def test_the_library_defaults_are_refused(self, two_arm_sim, tmp_path):
        _record(two_arm_sim, tmp_path, fps=30, name="multi_ds")
        assert self._run(two_arm_sim, 50.0)["status"] == "error"

    def test_the_refusal_names_the_method_both_rates_and_the_distortion(self, two_arm_sim, tmp_path):
        _record(two_arm_sim, tmp_path, fps=30, name="multi_named")
        text = self._run(two_arm_sim, 50.0)["content"][0]["text"]
        assert text.startswith("run_multi_policy: ")
        assert "declares 30 fps" in text
        assert "control_frequency=50 Hz" in text
        assert "1.667x" in text

    def test_the_envelope_is_the_shared_helper_verbatim(self, two_arm_sim, tmp_path):
        _record(two_arm_sim, tmp_path, fps=30, name="multi_verbatim")
        assert self._run(two_arm_sim, 50.0) == dataset_rate_mismatch_error(
            "run_multi_policy", two_arm_sim._active_recorder(), 50.0
        )

    def test_no_frame_is_written(self, two_arm_sim, tmp_path):
        root = _record(two_arm_sim, tmp_path, fps=30, name="multi_frames")
        self._run(two_arm_sim, 50.0)
        assert two_arm_sim._active_recorder().frame_count == 0
        assert _frames_on_disk(root) == 0

    def test_no_rollout_is_left_running(self, two_arm_sim, tmp_path):
        """The refusal precedes every per-robot rollout, so neither arm is driven."""
        _record(two_arm_sim, tmp_path, fps=30, name="multi_idle")
        self._run(two_arm_sim, 50.0)
        assert two_arm_sim._policy_threads == {}
        assert [two_arm_sim._world.robots[n].policy_running for n in ("armA", "armB")] == [False, False]

    def test_a_matching_rate_still_records_both_robots(self, two_arm_sim, tmp_path):
        """Control: at an aligned rate the merged capture is unaffected."""
        root = _record(two_arm_sim, tmp_path, fps=50, name="multi_aligned")
        assert self._run(two_arm_sim, 50.0)["status"] == "success"
        assert two_arm_sim.stop_recording()["status"] == "success"
        assert _frames_on_disk(root) == 10

    def test_a_rollout_with_no_recording_open_is_never_refused(self, two_arm_sim):
        """Control: with nothing to disagree with, any rate is usable."""
        assert self._run(two_arm_sim, 50.0)["status"] == "success"


_ENTRY_POINTS = {
    "strands_robots/simulation/base.py": ("run_policy", "eval_policy", "evaluate_benchmark"),
    "strands_robots/simulation/mujoco/simulation.py": ("start_policy", "run_multi_policy"),
}

# ``PolicyRunner`` is driven directly too, so it repeats the check under its own
# name. Kept as a separate table because the guard it calls differs: the engine
# returns a tool envelope, the runner raises for a caller that has no envelope.
_RUNNER_ENTRY_POINTS = {
    "strands_robots/simulation/policy_runner.py": ("run", "evaluate"),
}


def _self_calls(module: str, method: str) -> set[str]:
    """Names of the ``self.x(...)`` calls made anywhere inside ``module::method``."""
    tree = ast.parse(Path(module).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method:
            return {
                n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            }
    pytest.fail(f"{method} not found in {module}")


@pytest.mark.parametrize(
    ("module", "method"),
    [(m, f) for m, fns in _ENTRY_POINTS.items() for f in fns],
)
def test_every_rollout_entry_point_checks_the_recording_rate(module, method):
    """No rollout driver may capture into a dataset it disagrees with.

    Every driver in the table also returns its refusal in a behavioural case
    above, because a structural pin proves only that the guard is *called*:
    discarding the ``return`` while keeping the call still satisfies this sweep.
    What it is for is the next driver - one added later cannot quietly skip the
    guard the way ``run_multi_policy`` once skipped ``_validate_action_horizon``.
    """
    assert "_validate_recording_rate" in _self_calls(module, method), (
        f"{module}::{method} does not check the recording rate"
    )


@pytest.mark.parametrize(
    ("module", "method"),
    [(m, f) for m, fns in _RUNNER_ENTRY_POINTS.items() for f in fns],
)
def test_every_directly_drivable_runner_method_checks_the_recording_rate(module, method):
    """The runner cannot rely on a guard that is not on a direct caller's path."""
    assert "_reject_recording_rate_mismatch" in _self_calls(module, method), (
        f"{module}::{method} does not check the recording rate"
    )
